import os
import re
import time
import random
import numpy as np
import cv2
from logzero import logger as log
import torch
import torch.nn as nn
from torch.functional import F
import torch.utils.data
from torch.optim import lr_scheduler
from .data_loader import InpaintorLoader
from src.models.inpaintor import Inpaintor
from src.models.dis import InpaintDiscriminator


def get_scheduler(optimizer, train_epoch_num=400):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 0 + 1 - 5) / float(train_epoch_num + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


############################################################
#  DataParallel_withLoss
############################################################
class FullModel_dis(nn.Module):
    def __init__(self, inpaint, dis):
        super(FullModel_dis, self).__init__()
        self.inpaint = inpaint
        self.dis = dis

    def hinge_loss(self, real, fake):
        return (F.relu(1 + real) + F.relu(1 - fake)).mean()

    def forward(self, inputs):
        img_norm = inputs[0].cuda().float()
        bgimg = inputs[1].cuda().float()
        img_smooth = inputs[2].cuda().float()
        alpha = inputs[3].cuda().float()
        img_size_for_crop = 256
        crop_x = random.randint(0, img_size_for_crop - 128)
        crop_y = random.randint(0, img_size_for_crop - 128)

        with torch.no_grad():
            inpaint_out, _ = self.inpaint(img_norm, alpha)
        d_fake = self.dis(inpaint_out)
        inpaint_out_crop = inpaint_out[:, :,
                                       crop_y:crop_y + 128, crop_x:crop_x + 128]
        d_fake_crop = self.dis(inpaint_out_crop)

        bgimg = F.interpolate(
            bgimg,
            (img_size_for_crop,
             img_size_for_crop),
            mode='bilinear')
        img_smooth = F.interpolate(
            img_smooth,
            (img_size_for_crop,
             img_size_for_crop),
            mode='bilinear')

        bgimg_crop = bgimg[:, :, crop_y:crop_y + 128, crop_x:crop_x + 128]
        img_smooth_crop = img_smooth[:, :,
                                     crop_y:crop_y + 128, crop_x:crop_x + 128]
        d_real = self.dis(bgimg)
        d_real_crop = self.dis(bgimg_crop)
        d_loss_global = self.hinge_loss(d_real, d_fake)
        d_loss_global_crop = self.hinge_loss(d_real_crop, d_fake_crop)
        d_loss = d_loss_global + d_loss_global_crop

        return torch.unsqueeze(d_loss, 0)


class FullModel_gen(nn.Module):
    def __init__(self, inpaint, dis):
        super(FullModel_gen, self).__init__()
        self.inpaint = inpaint
        self.dis = dis

    def hinge_loss(self, real, fake):
        return (F.relu(1 + real) + F.relu(1 - fake)).mean()

    def forward(self, inputs):
        img_norm = inputs[0].cuda().float()
        bgimg = inputs[1].cuda().float()
        img_smooth = inputs[2].cuda().float()
        alpha = inputs[3].cuda().float()
        img_size_for_crop = 256
        crop_x = random.randint(0, img_size_for_crop - 128)
        crop_y = random.randint(0, img_size_for_crop - 128)

        img_size_for_crop_256 = random.randint(128, 256)
        crop_x_256 = random.randint(0, img_size_for_crop_256 - 128)
        crop_y_256 = random.randint(0, img_size_for_crop_256 - 128)

        inpaint_out, de2im = self.inpaint(img_norm, alpha)

        de2im_loss0 = F.l1_loss(de2im[0], F.interpolate(
            bgimg, de2im[0].shape[2:4], mode='bilinear'))
        de2im_loss1 = F.l1_loss(de2im[1], F.interpolate(
            img_smooth, de2im[1].shape[2:4], mode='bilinear'))
        de2im_loss = de2im_loss0 + de2im_loss1
        inp_loss = F.l1_loss(inpaint_out, F.interpolate(
            bgimg, inpaint_out.shape[2:4], mode='bilinear'))
        d_fake = self.dis(inpaint_out)
        inpaint_out_crop = inpaint_out[:,
                                       :,
                                       crop_y_256:crop_y_256 + 128,
                                       crop_x_256:crop_x_256 + 128]
        d_fake_crop = self.dis(inpaint_out_crop)
        g_loss = torch.mean(d_fake.mean(dim=1)) + \
            torch.mean(d_fake_crop.mean(dim=1))
        return (inpaint_out, de2im,
                torch.unsqueeze(inp_loss, 0), torch.unsqueeze(de2im_loss, 0),
                torch.unsqueeze(g_loss, 0))

############################################################
#  Trainer
############################################################


class Trainer():
    def __init__(
            self,
            inpaintor,
            discriminator,
            data_list,
            save_data_path,
            model_id,
            text_pool_num=10,
            batch_size=1,
            nworker=2):
        super(Trainer, self).__init__()
        self.inpaintor = inpaintor
        self.discriminator = discriminator
        self.dataset = InpaintorLoader(data_list, text_pool_num=text_pool_num)
        self.savedir = os.path.join(save_data_path, model_id)
        self.layer_regex = {
            "generative": r"(encoder.*)|(PCblock.*)|(decoder.*)",
            "dis": r"(model.*)|(to_logits.*)",
        }
        self.epoch = 0
        self.batch_size = batch_size
        self.nworker = nworker
        self.train_epoch_num = 1000
        self.param_generatice = [
            param for name,
            param in inpaintor.named_parameters() if bool(
                re.fullmatch(
                    self.layer_regex["generative"],
                    name))]
        self.param_dis = [
            param for name,
            param in discriminator.named_parameters() if bool(
                re.fullmatch(
                    self.layer_regex["dis"],
                    name))]
        self.optimizer_gen = torch.optim.Adam(
            self.param_generatice, lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_dis = torch.optim.Adam(
            self.param_dis, lr=0.0002, betas=(0.5, 0.999))
        self.fullmodel_dis = FullModel_dis(self.inpaintor, self.discriminator)
        self.fullmodel_dis = nn.DataParallel(self.fullmodel_dis)
        self.fullmodel_gen = FullModel_gen(self.inpaintor, self.discriminator)
        self.fullmodel_gen = nn.DataParallel(self.fullmodel_gen)
        # for multi gpu training
        self.schedulers = []
        self.schedulers.append(
            get_scheduler(
                self.optimizer_gen,
                self.train_epoch_num))
        self.schedulers.append(
            get_scheduler(
                self.optimizer_dis,
                self.train_epoch_num))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def train_model(self):
        epochs = self.train_epoch_num
        # Data generators
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.nworker,
            pin_memory=True)
        self.inpaintor.train()
        self.discriminator.train()
        for epoch in range(0, epochs):
            log.debug("Epoch {}/{}.".format(epoch, epochs))
            self.epoch = epoch
            # Training
            self.train_epoch(dataloader)
            if (epoch % 5 == 0):
                torch.cuda.empty_cache()
                torch.save(
                    self.inpaintor.state_dict(),
                    os.path.join(
                        self.savedir,
                        'inp_{}.pth'.format(epoch)))
                torch.save(
                    self.discriminator.state_dict(),
                    os.path.join(
                        self.savedir,
                        'dis_{}.pth'.format(epoch)))
                torch.save(
                    self.optimizer_gen.state_dict(),
                    os.path.join(
                        self.savedir,
                        'optg_{}.pth'.format(epoch)))
                torch.save(
                    self.optimizer_dis.state_dict(),
                    os.path.join(
                        self.savedir,
                        'optd_{}.pth'.format(epoch)))
                torch.cuda.empty_cache()
            self.update_learning_rate()

    def train_epoch(self, dataloader):
        self.steps = len(dataloader)
        self.step = 0
        self.cnt = 0
        end = time.time()
        for inputs in dataloader:
            self.train_step(inputs, end)
            end = time.time()
            self.step += 1

    def train_step(self, inputs, end):
        start = time.time()
        loss_dis = self.fullmodel_dis(inputs)
        loss = torch.mean(loss_dis)
        self.optimizer_dis.zero_grad()
        loss.backward()
        self.optimizer_dis.step()

        inpaint_out, de2im, inp_loss, de2im_loss, loss_gen = self.fullmodel_gen(
            inputs)
        inp_loss = torch.mean(inp_loss)
        de2im_loss = torch.mean(de2im_loss)
        loss_gen = torch.mean(loss_gen)
        loss = inp_loss + de2im_loss + loss_gen * 0.2
        self.optimizer_gen.zero_grad()
        loss.backward()
        self.optimizer_gen.step()

        forward_time = time.time()
        if (self.step % 100 == 0):
            log.debug(self.step, self.epoch)
            torch.cuda.empty_cache()
            self.save_contents(inputs, inpaint_out, de2im)
            torch.cuda.empty_cache()
        if (self.step % 100 == 0):
            data_show = "{}/{}/{}/{}, forward_time: {:.3f} data {:.3f}".format(
                self.epoch, self.cnt, self.step + 1, self.steps, forward_time - start, (start - end))
            log.debug(data_show)
            self.show_loss(
                torch.mean(inp_loss).item(),
                torch.mean(de2im_loss).item(),
                torch.mean(loss_gen).item(),
                torch.mean(loss_dis).item())

    def show_loss(self, inp_loss, de2im_loss, loss_gen, loss_dis):
        inp_loss_show = "inp_loss: {:.3f} de2im_loss: {:.3f} loss_gen: {:.3f}".format(
            inp_loss, de2im_loss, loss_gen)
        dis_loss_show = "loss_dis_global: {:.3f}".format(loss_dis)
        log.debug(inp_loss_show)
        log.debug(dis_loss_show)

    def save_contents(self, inputs, text_inpaint_pred, de2im):
        # inputs
        img_norm, bgimg, img_smooth, alpha = inputs

        sn = os.path.join(
            self.savedir, 'i_{}_{}.jpg'.format(self.epoch, self.cnt))
        img_save = (
            img_norm.data.cpu().numpy()[0].transpose(
                1, 2, 0)[
                :, :, ::-1] * 0.5 + 0.5) * 255
        cv2.imwrite(sn, img_save)
        sn = os.path.join(
            self.savedir, 'is_{}_{}.jpg'.format(self.epoch, self.cnt))
        img_save = (
            img_smooth.data.cpu().numpy()[0].transpose(
                1, 2, 0)[
                :, :, ::-1] * 0.5 + 0.5) * 255
        cv2.imwrite(sn, img_save)
        sn = os.path.join(
            self.savedir, 'bg_{}_{}.jpg'.format(self.epoch, self.cnt))
        bgimg_save = (
            bgimg.data.cpu().numpy()[0].transpose(
                1, 2, 0)[
                :, :, ::-1] * 0.5 + 0.5) * 255
        bgimg_saver = bgimg_save.copy()
        cv2.imwrite(sn, bgimg_save)

        sn = os.path.join(
            self.savedir,
            'inp0_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        inpimg_save = (
            text_inpaint_pred.data.cpu().numpy()[0].transpose(
                1, 2, 0)[
                :, :, ::-1] * 0.5 + 0.5) * 255
        inpimg_save = cv2.resize(inpimg_save, (256, 256))
        cv2.imwrite(sn, inpimg_save)
        sn = os.path.join(
            self.savedir,
            'alpha_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        inpimg_save = (alpha.data.cpu().numpy()[0, 0]) * 255
        inpimg_save = cv2.resize(inpimg_save, (256, 256))
        cv2.imwrite(sn, inpimg_save)
        sn = os.path.join(
            self.savedir,
            'inpd0_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        inpimg_save = de2im[0].data.cpu().numpy()[0].transpose(1, 2, 0)[
            :, :, ::-1] * 0.5 + 0.5
        inpimg_save = (cv2.resize(inpimg_save, (256, 256),
                                  interpolation=cv2.INTER_NEAREST)) * 255
        cv2.imwrite(sn, inpimg_save)
        sn = os.path.join(
            self.savedir,
            'inpd1_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        inpimg_save = de2im[1].data.cpu().numpy()[0].transpose(1, 2, 0)[
            :, :, ::-1] * 0.5 + 0.5
        inpimg_save = (cv2.resize(inpimg_save, (256, 256),
                                  interpolation=cv2.INTER_NEAREST)) * 255
        cv2.imwrite(sn, inpimg_save)

        self.cnt += 1


def train(args, data_list):
    if args.gpuid == -1:
        dev = torch.device(f"cuda")
    else:
        dev = torch.device(f"cuda:{args.gpuid}")
    inpaintor = Inpaintor(dev=dev)
    dis = InpaintDiscriminator()
    inpaintor.to(dev)
    save_data_path = args.save_data_path
    model_id = args.model_id
    model_trainer = Trainer(inpaintor,
                            dis,
                            data_list,
                            save_data_path,
                            model_id,
                            batch_size=args.batch_size,
                            nworker=args.nworker
                            ).train_model()
