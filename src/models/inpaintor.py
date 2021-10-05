import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dto.dto_model import TextInfo
from .layers.inner_inpaint import (
    BASE,
    ConvDown,
    ConvUp,
    PCBActiv,
    ResnetBlock,
    UnetSkipConnectionDBlock,
    UnetSkipConnectionEBlock,
)


class InnerCos(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_model = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0), nn.Tanh()
        )

    def forward(self, inp):
        inp0, inp1 = inp
        out0 = self.down_model(inp0)
        out1 = self.down_model(inp1)
        return out0, out1


class PCconv(nn.Module):
    def __init__(self, dev: torch.device = None):
        super().__init__()
        self.down_128 = ConvDown(64, 128, 4, 2, padding=1, layers=2)
        self.down_64 = ConvDown(128, 256, 4, 2, padding=1)
        self.down_32 = ConvDown(256, 256, 1, 1)
        self.down_16 = ConvDown(512, 512, 4, 2, padding=1, activ=False)
        self.down_8 = ConvDown(
            512,
            512,
            4,
            2,
            padding=1,
            layers=2,
            activ=False)
        self.down_4 = ConvDown(
            512,
            512,
            4,
            2,
            padding=1,
            layers=3,
            activ=False)
        self.down = ConvDown(768, 256, 1, 1)
        self.fuse = ConvDown(512, 512, 1, 1)
        self.up = ConvUp(512, 256, 1, 1)
        self.up_128 = ConvUp(512, 64, 1, 1)
        self.up_64 = ConvUp(512, 128, 1, 1)
        self.up_32 = ConvUp(512, 256, 1, 1)
        self.base = BASE(512, dev=dev)
        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for _ in range(5):
            seuqence_3 += [PCBActiv(256, 256, innorm=True)]
            seuqence_5 += [PCBActiv(256, 256, sample="same-5", innorm=True)]
            seuqence_7 += [PCBActiv(256, 256, sample="same-7", innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, mask):
        # mask = util.cal_feat_mask(mask, 3)
        # input[2]:256 32 32
        b, c, h, w = input[2].size()
        # mask_1 = torch.add(torch.neg(mask.float()), 1)
        # mask_1 = mask_1.expand(b, c, h, w)
        mask_1 = mask

        x_1 = self.activation(input[0])
        x_2 = self.activation(input[1])
        x_3 = self.activation(input[2])
        x_4 = self.activation(input[3])
        x_5 = self.activation(input[4])
        x_6 = self.activation(input[5])
        # Change the shape of each layer and intergrate low-level/high-level
        # features
        x_1 = self.down_128(x_1)
        x_2 = self.down_64(x_2)
        x_3 = self.down_32(x_3)
        x_4 = self.up(x_4, (32, 32))
        x_5 = self.up(x_5, (32, 32))
        x_6 = self.up(x_6, (32, 32))

        # The first three layers are Texture/detail
        # The last three layers are Structure
        x_DE = torch.cat([x_1, x_2, x_3], 1)
        x_ST = torch.cat([x_4, x_5, x_6], 1)

        x_ST = self.down(x_ST)
        x_DE = self.down(x_DE)
        x_ST = [x_ST, mask_1]
        x_DE = [x_DE, mask_1]

        # Multi Scale PConv fill the Details
        x_DE_3 = self.cov_3(x_DE)
        x_DE_5 = self.cov_5(x_DE)
        x_DE_7 = self.cov_7(x_DE)
        x_DE_fuse = torch.cat([x_DE_3[0], x_DE_5[0], x_DE_7[0]], 1)
        x_DE_fi = self.down(x_DE_fuse)

        # Multi Scale PConv fill the Structure
        x_ST_3 = self.cov_3(x_ST)
        x_ST_5 = self.cov_5(x_ST)
        x_ST_7 = self.cov_7(x_ST)
        x_ST_fuse = torch.cat([x_ST_3[0], x_ST_5[0], x_ST_7[0]], 1)
        x_ST_fi = self.down(x_ST_fuse)

        x_cat = torch.cat([x_ST_fi, x_DE_fi], 1)
        x_cat_fuse = self.fuse(x_cat)

        # Feature equalizations
        x_final = self.base(x_cat_fuse)

        # Add back to the input
        x_ST = x_final
        x_DE = x_final
        x_1 = self.up_128(x_DE, (128, 128)) + input[0]
        x_2 = self.up_64(x_DE, (64, 64)) + input[1]
        x_3 = self.up_32(x_DE, (32, 32)) + input[2]
        x_4 = self.down_16(x_ST) + input[3]
        x_5 = self.down_8(x_ST) + input[4]
        x_6 = self.down_4(x_ST) + input[5]

        out = [x_1, x_2, x_3, x_4, x_5, x_6]
        loss = [x_ST_fi, x_DE_fi]
        out_final = [out, loss]
        return out_final


class Encoder(nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=64 * 8,
        ngf=64,
        res_num=4,
        norm_layer=nn.InstanceNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        # construct unet structure
        Encoder_1 = UnetSkipConnectionEBlock(
            input_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            outermost=True,
        )
        Encoder_2 = UnetSkipConnectionEBlock(
            ngf, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_3 = UnetSkipConnectionEBlock(
            ngf * 2, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_4 = UnetSkipConnectionEBlock(
            ngf * 4, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_5 = UnetSkipConnectionEBlock(
            ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Encoder_6 = UnetSkipConnectionEBlock(
            ngf * 8,
            ngf * 8,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            innermost=True,
        )
        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.Encoder_1 = Encoder_1
        self.Encoder_2 = Encoder_2
        self.Encoder_3 = Encoder_3
        self.Encoder_4 = Encoder_4
        self.Encoder_5 = Encoder_5
        self.Encoder_6 = Encoder_6

    def forward(self, input):
        y_1 = self.Encoder_1(input)
        y_2 = self.Encoder_2(y_1)
        y_3 = self.Encoder_3(y_2)
        y_4 = self.Encoder_4(y_3)
        y_5 = self.Encoder_5(y_4)
        y_6 = self.Encoder_6(y_5)
        y_7 = self.middle(y_6)

        return y_1, y_2, y_3, y_4, y_5, y_7


class Decoder(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.InstanceNorm2d,
        use_dropout=False,
    ):
        super().__init__()

        # construct unet structure
        Decoder_1 = UnetSkipConnectionDBlock(
            ngf * 8,
            ngf * 8,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            innermost=True,
        )
        Decoder_2 = UnetSkipConnectionDBlock(
            ngf * 16, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_3 = UnetSkipConnectionDBlock(
            ngf * 16, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_4 = UnetSkipConnectionDBlock(
            ngf * 8, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_5 = UnetSkipConnectionDBlock(
            ngf * 4, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
        Decoder_6 = UnetSkipConnectionDBlock(
            ngf * 2,
            output_nc,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            outermost=True,
        )

        self.Decoder_1 = Decoder_1
        self.Decoder_2 = Decoder_2
        self.Decoder_3 = Decoder_3
        self.Decoder_4 = Decoder_4
        self.Decoder_5 = Decoder_5
        self.Decoder_6 = Decoder_6

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        y_1 = self.Decoder_1(input_6)
        y_2 = self.Decoder_2(torch.cat([y_1, input_5], 1))
        y_3 = self.Decoder_3(torch.cat([y_2, input_4], 1))
        y_4 = self.Decoder_4(torch.cat([y_3, input_3], 1))
        y_5 = self.Decoder_5(torch.cat([y_4, input_2], 1))
        y_6 = self.Decoder_6(torch.cat([y_5, input_1], 1))
        out = y_6

        return out


class PCblock(nn.Module):
    def __init__(self, dev: torch.device = None):
        super().__init__()
        self.pc_block = PCconv(dev=dev)
        innerloss = InnerCos()
        loss = [innerloss]
        self.loss = nn.Sequential(*loss)

    def forward(self, input, mask):
        out = self.pc_block(input, mask)
        de = self.loss(out[1])
        return out[0], de


class Inpaintor(nn.Module):
    def __init__(
        self,
        alpha_threshold_for_mask: float = 0.1,
        text_fg_threshold_for_mask: float = 0.25,
        inpaint_img_size: int = 256,
        dev: torch.device = None,
    ):
        super().__init__()
        self.encoder = Encoder(6, 64)
        self.PCblock = PCblock(dev=dev)
        self.decoder = Decoder(3, 3)
        self.alpha_threshold_for_mask = alpha_threshold_for_mask
        self.text_fg_threshold_for_mask = text_fg_threshold_for_mask
        self.inpaint_img_size = inpaint_img_size
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def getmask(self, size, channel, alpha, text_fg=None):
        alpha = F.interpolate(alpha, (size, size), mode="bilinear").detach()
        if text_fg is not None:
            text_fg = F.interpolate(
                text_fg, (size, size), mode="bilinear").detach()
            text_mask = (text_fg[:, 1, :, :] > self.text_fg_threshold_for_mask) & (
                (alpha[:, 0, :, :] > self.alpha_threshold_for_mask)
                | (alpha[:, 1, :, :] > self.alpha_threshold_for_mask)
                | (alpha[:, 2, :, :] > self.alpha_threshold_for_mask)
            )
        else:
            text_mask = (
                (alpha[:, 0, :, :] > self.alpha_threshold_for_mask)
            )
        text_mask = text_mask.unsqueeze(1).float()
        text_mask = text_mask.expand(text_mask.shape[0], channel, size, size)
        return text_mask

    def get_hole_image(self, img, mask_256, mask_32):
        # image initialization with mask
        img.narrow(1, 0, 1).masked_fill_(
            mask_256.narrow(1, 0, 1).type(torch.bool), 2 * 123.0 / 255.0 - 1.0
        )
        img.narrow(1, 1, 1).masked_fill_(
            mask_256.narrow(1, 0, 1).type(torch.bool), 2 * 104.0 / 255.0 - 1.0
        )
        img.narrow(1, 2, 1).masked_fill_(
            mask_256.narrow(1, 0, 1).type(torch.bool), 2 * 117.0 / 255.0 - 1.0
        )
        mask_32 = torch.add(torch.neg(mask_32.float()), 1).float()
        mask_256 = torch.add(torch.neg(mask_256.float()), 1).float()
        inp = torch.cat((img, mask_256), 1)
        return inp, mask_32

    def preprocessing_test(self, img, ti: TextInfo):
        word_out, char_out, recog_out = ti.ocr_outs
        text_fg_pred, text_tblr_pred, text_orient_pred = word_out
        char_fg_pred, char_tblr_pred, char_orient_pred = char_out
        text_fg_pred = F.softmax(text_fg_pred, 1)
        # mask for inpainting
        mask_256 = self.getmask(
            self.inpaint_img_size,
            3,
            ti.alpha_outs,
            text_fg_pred)
        mask_32 = self.getmask(
            self.inpaint_img_size // 8,
            256,
            ti.alpha_outs,
            text_fg_pred)
        inp, mask_32 = self.get_hole_image(img, mask_256, mask_32)
        return inp, mask_32

    def preprocessing_train(self, img, alpha: torch.Tensor):
        mask_256 = self.getmask(self.inpaint_img_size, 3, alpha)
        mask_32 = self.getmask(self.inpaint_img_size // 8, 256, alpha)
        inp, mask_32 = self.get_hole_image(img, mask_256, mask_32)
        return inp, mask_32

    def forward(self, img, mask_info):
        img = F.interpolate(
            img,
            (self.inpaint_img_size,
             self.inpaint_img_size),
            mode="bilinear")
        if self.training:
            inp, mask_32 = self.preprocessing_train(img, mask_info)
            fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6 = self.encoder(
                inp)
            De_in = [
                fake_p_1,
                fake_p_2,
                fake_p_3,
                fake_p_4,
                fake_p_5,
                fake_p_6]
            x_out, de_out = self.PCblock(De_in, mask_32)
            out = self.decoder(
                x_out[0],
                x_out[1],
                x_out[2],
                x_out[3],
                x_out[4],
                x_out[5])
            return out, de_out
        else:
            inp, mask_32 = self.preprocessing_test(img, mask_info)
            fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6 = self.encoder(
                inp)
            De_in = [
                fake_p_1,
                fake_p_2,
                fake_p_3,
                fake_p_4,
                fake_p_5,
                fake_p_6]
            x_out, de_out = self.PCblock(De_in, mask_32)
            out = self.decoder(
                x_out[0],
                x_out[1],
                x_out[2],
                x_out[3],
                x_out[4],
                x_out[5])
            return out
