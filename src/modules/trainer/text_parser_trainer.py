import os
import re
import time
import numpy as np
import cv2
from logzero import logger as log
import torch
import torch.nn as nn
from torch.functional import F
import torch.utils.data
from . import loss
from .data_loader import TextParserLoader
from torch.optim import lr_scheduler
from src.models.vectorization import Vectorization


def get_scheduler(optimizer, all_step_num):
    def lambda_rule(step):
        rate = float(step) / all_step_num

        def cosine_rampdown(rate):
            """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
            return float(.5 * (np.cos(np.pi * rate) + 1))
        lr_l = cosine_rampdown(rate)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

############################################################
#  DataParallel_withLoss
############################################################


class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs):
        img_norm = inputs[0].cuda().float()
        img_org = inputs[1].cuda().float()
        (text_level_data, char_level_data) = inputs[2]
        (text_fg_mask, text_ins_mask, text_scoremap, text_scoremap_qt,
         text_geomap, text_rectangles_array) = text_level_data
        text_ins_mask = text_ins_mask.cuda().float()
        outputs = self.model(img_norm, img_org, text_ins_mask)
        total_loss, loss_items = self.loss(inputs, outputs)
        textinfo, _ = outputs
        text_preds, char_preds, recognition_results = textinfo.ocr_outs
        text_fg_pred, text_tblr_pred, text_orient_pred = text_preds
        char_fg_pred, char_tblr_pred, char_orient_pred = char_preds
        outs = text_fg_pred, char_fg_pred, textinfo.alpha_outs
        return (torch.unsqueeze(total_loss, 0),
                outs,
                loss_items)

############################################################
#  Trainer
############################################################


class Trainer():
    def __init__(
            self,
            model,
            data_list,
            save_data_path,
            model_id,
            text_pool_num=10,
            batch_size=1,
            loss_func=None):
        super(Trainer, self).__init__()
        self.model = model
        self.dataset = TextParserLoader(data_list, text_pool_num=text_pool_num)
        self.batch_size = batch_size
        self.savedir = os.path.join(save_data_path, model_id)
        os.makedirs(self.savedir, exist_ok=True)
        self.layer_regex = {
            "lr1": r"(backbone.*)",
            "lr10": r"(text_parser.*)|(down.*)",
        }
        self.epoch = 0
        self.param_lr_1x = [
            param for name,
            param in model.named_parameters() if bool(
                re.fullmatch(
                    self.layer_regex["lr1"],
                    name))]
        self.param_lr_10x = [
            param for name,
            param in model.named_parameters() if bool(
                re.fullmatch(
                    self.layer_regex["lr10"],
                    name))]
        self.loss = loss.TextParserLossFunc()
        self.fullmodel = FullModel(self.model, self.loss)
        # for multi gpu training
        self.fullmodel = nn.DataParallel(self.fullmodel)
        lr = 1e-2
        self.optimizer = torch.optim.SGD([
            {'params': self.param_lr_1x, 'lr': lr * 1, 'weight_decay': 2e-4},
            {'params': self.param_lr_10x, 'lr': lr * 10, 'weight_decay': 2e-4},
        ], lr=lr, momentum=0.9, weight_decay=2e-4)
        self.epochs = 31
        # for computing the number of step
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True)
        all_step_num = len(dataloader) * self.epochs
        self.scheduler = get_scheduler(
            self.optimizer, all_step_num=all_step_num)

    def train_model(self):
        # epochs=self.config.EPOCHS
        epochs = self.epochs
        self.LR_RAMPDOWN_EPOCHS = int(epochs)
        # Data generators
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True)
        self.model.train()
        for epoch in range(0, epochs):
            log.debug(f"Epoch {epoch}/{epochs}")
            self.epoch = epoch
            # Training
            self.train_epoch(dataloader)
            if (epoch % 5 == 0):
                torch.cuda.empty_cache()
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.savedir,
                        'model_{}.pth'.format(epoch)))
                torch.cuda.empty_cache()

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
        total_loss, outputs, loss_items = self.fullmodel(inputs)

        total_loss = torch.mean(total_loss)
        forward_time = time.time()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if (self.step % 200 == 0):
            log.debug(f"{self.step},{self.epoch}")
            torch.cuda.empty_cache()
        if (self.step % 200 == 0):
            torch.cuda.empty_cache()
            data_show = "{}/{}/{}, forward_time: {:.3f} data {:.3f}".format(
                self.epoch, self.step + 1, self.steps, forward_time - start, (start - end))
            log.debug(data_show)
            self.show_loss(loss_items)
            self.save_materials(inputs, outputs)
            torch.cuda.empty_cache()

    def show_loss(self, loss_items):
        (ocr_loss_items, param_loss_items, alpha_losses, font_loss) = loss_items
        shadow_loss, stroke_loss = param_loss_items
        (stroke_param_loss, stroke_visibility_loss) = stroke_loss
        (shadow_param_loss, shadow_visibility_loss) = shadow_loss
        (ocr_loss,
         text_fg_loss, char_fg_loss, char_cls_loss,
         text_geo_loss, text_angle_loss,
         char_geo_loss, char_angle_loss) = ocr_loss_items
        fill_alpha_fg_loss, fill_alpha_bg_loss, shadow_alpha_fg_loss, shadow_alpha_bg_loss, \
            stroke_alpha_fg_loss, stroke_alpha_bg_loss = alpha_losses

        seg_loss_show = "ocr loss: {:.3f} text_fg_loss: {:.3f} char_fg_loss: {:.3f} char_cls_loss: {:.3f}".format(
            torch.mean(ocr_loss).item(), torch.mean(text_fg_loss).item(),
            torch.mean(char_fg_loss).item(), torch.mean(char_cls_loss).item())
        geo_font_loss_show = "text_geo_loss: {:.3f} text_angle_loss: {:.3f} char_geo_loss: {:.3f} font_loss: {:.3f}".format(torch.mean(
            text_geo_loss).item(), torch.mean(text_angle_loss).item(), torch.mean(char_geo_loss).item(), torch.mean(font_loss).item())
        shadow_param_loss_show = "shadow_param_loss: {:.3f} shadow_visibility_loss: {:.3f}".format(
            torch.mean(shadow_param_loss).item(), torch.mean(shadow_visibility_loss).item(), )
        stroke_param_loss_show = "stroke_param_loss: {:.3f} stroke_visibility_loss: {:.3f}".format(
            torch.mean(stroke_param_loss).item(), torch.mean(stroke_visibility_loss).item(), )
        alpha_fg_loss_show = "fg alpha_loss0: {:.3f} alpha_loss1: {:.3f} alpha_loss2: {:.3f}".format(
            torch.mean(fill_alpha_fg_loss).item(),
            torch.mean(shadow_alpha_fg_loss).item(),
            torch.mean(stroke_alpha_fg_loss).item(),
        )
        alpha_bg_loss_show = "bg alpha_loss0: {:.3f} alpha_loss1: {:.3f} alpha_loss2: {:.3f}".format(
            torch.mean(fill_alpha_bg_loss).item(),
            torch.mean(shadow_alpha_bg_loss).item(),
            torch.mean(stroke_alpha_bg_loss).item(),
        )
        log.debug(f"{seg_loss_show}")
        log.debug(f"{geo_font_loss_show}")
        log.debug(f"{shadow_param_loss_show}")
        log.debug(f"{stroke_param_loss_show}")
        log.debug(f"{alpha_fg_loss_show}")
        log.debug(f"{alpha_bg_loss_show}")

    def save_materials(self, inputs, outputs):
        start = time.time()
        # inputs
        img_norm, img_org, ocr_data, style_data = inputs
        (alpha, shadow_data, stroke_data, font_label, valid_text_index) = style_data
        text_level_data, char_level_data = ocr_data
        text_fg_mask, text_ins_mask, text_scoremap, text_scoremap_qt, text_geomap, text_rectangles_array = text_level_data
        char_fg_mask, char_ins_mask, char_scoremap, char_scoremap_qt, char_geomap, char_cls_mask, char_rectangles_array = char_level_data

        text_fg_pred, char_fg_pred, alpha_outs = outputs

        sn = os.path.join(
            self.savedir, 'i_{}_{}.jpg'.format(self.epoch, self.cnt))
        img_save = (
            img_org[0].data.cpu().numpy().transpose(
                1, 2, 0)[
                :, :, ::-1] * 0.5 + 0.5) * 255
        cv2.imwrite(sn, img_save)

        sn = os.path.join(
            self.savedir,
            'fillp_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        cv2.imwrite(sn, alpha_outs[0, 0, :, :].data.cpu().numpy() * 255)
        sn = os.path.join(
            self.savedir,
            'fillg_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        cv2.imwrite(sn, alpha[0, 0, :, :].data.cpu().numpy() * 255)
        sn = os.path.join(
            self.savedir,
            'strokep_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        cv2.imwrite(sn, alpha_outs[0, 1, :, :].data.cpu().numpy() * 255)
        sn = os.path.join(
            self.savedir,
            'strokeg_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        cv2.imwrite(sn, alpha[0, 1, :, :].data.cpu().numpy() * 255)
        sn = os.path.join(
            self.savedir,
            'shadowp_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        cv2.imwrite(sn, alpha_outs[0, 2, :, :].data.cpu().numpy() * 255)
        sn = os.path.join(
            self.savedir,
            'shadowg_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        cv2.imwrite(sn, alpha[0, 2, :, :].data.cpu().numpy() * 255)

        sn = os.path.join(
            self.savedir,
            'tfp_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        text_fg_pred = F.softmax(text_fg_pred, 1)
        cv2.imwrite(sn, text_fg_pred[0, 1].data.cpu().numpy() * 255)

        sn = os.path.join(
            self.savedir,
            'tfg_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        text_fg_mask = text_fg_mask[0].data.cpu().numpy()
        text_fg_mask[text_fg_mask == 255] = 0.5
        cv2.imwrite(sn, text_fg_mask * 255)

        sn = os.path.join(
            self.savedir,
            'cfp_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        char_fg_pred = F.softmax(char_fg_pred, 1)
        cv2.imwrite(sn, char_fg_pred[0, 1].data.cpu().numpy() * 255)

        sn = os.path.join(
            self.savedir,
            'cfg_{}_{}.jpg'.format(
                self.epoch,
                self.cnt))
        char_fg_mask = char_scoremap[0].data.cpu().numpy()
        char_fg_mask[char_fg_mask == 255] = 0.5
        cv2.imwrite(sn, char_fg_mask * 255)
        self.cnt += 1


def train(args, data_list):
    if args.gpuid == -1:
        dev = torch.device(f"cuda")
    else:
        dev = torch.device(f"cuda:{args.gpuid}")
    model = Vectorization(text_pool_num=args.text_pool_num, dev=dev)
    model.to(dev)
    wfile = torch.load(args.pret)
    model.load_state_dict(wfile, strict=False)
    save_data_path = args.save_data_path
    model_id = args.model_id
    model_trainer = Trainer(model,
                            data_list,
                            save_data_path,
                            model_id,
                            batch_size=args.batch_size
                            ).train_model()
