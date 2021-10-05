import math

import numpy as np
import torch
import torch.nn as nn
from torch.functional import F


class PartialConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, inputt):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        input = inputt[0]
        mask = inputt[1]

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(
                1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        out = []
        out.append(output)
        out.append(new_mask)
        return out


class PCBActiv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        bn=True,
        sample="none-3",
        activ="leaky",
        conv_bias=False,
        innorm=False,
        inner=False,
        outer=False,
    ):
        super().__init__()
        if sample == "same-5":
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == "same-7":
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == "down-3":
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)
        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        elif self.innorm:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        elif self.outer:
            out = self.conv(out)
            out[0] = self.bn(out[0])
        else:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(
                channel,
                channel //
                reduction,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.ReLU(
                inplace=True),
            nn.Conv2d(
                channel //
                reduction,
                channel,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)


def gussin(v):
    outk = []
    v = v
    size = 32
    for i in range(size):
        for k in range(size):

            out = []
            for x in range(size):
                row = []
                for y in range(size):
                    cord_x = i
                    cord_y = k
                    dis_x = np.abs(x - cord_x)
                    dis_y = np.abs(y - cord_y)
                    dis_add = -(dis_x * dis_x + dis_y * dis_y)
                    dis_add = dis_add / (2 * v * v)
                    dis_add = math.exp(dis_add) / (2 * math.pi * v * v)

                    row.append(dis_add)
                out.append(row)

            outk.append(out)

    out = np.array(outk)
    f = out.sum(-1).sum(-1)
    q = []
    for i in range(1024):
        g = out[i] / f[i]
        q.append(g)
    out = np.array(q)
    return torch.from_numpy(out)


class Selfpatch(object):
    def buildAutoencoder(
        self, target_img, target_img_2, target_img_3, patch_size=1, stride=1
    ):
        nDim = 3
        assert target_img.dim() == nDim, "target image must be of dimension 3."

        self.Tensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor
        )

        patches_features = self._extract_patches(
            target_img, patch_size, stride)
        patches_features_f = self._extract_patches(
            target_img_3, patch_size, stride)

        patches_on = self._extract_patches(target_img_2, 1, stride)

        return patches_features_f, patches_features, patches_on

    def build(self, target_img, patch_size=5, stride=1):
        nDim = 3
        assert target_img.dim() == nDim, "target image must be of dimension 3."

        self.Tensor = (
            torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor
        )

        patches_features = self._extract_patches(
            target_img, patch_size, stride)

        return patches_features

    def _build(
        self,
        patch_size,
        stride,
        C,
        target_patches,
        npatches,
        normalize,
        interpolate,
        type,
    ):
        # for each patch, divide by its L2 norm.
        if type == 1:
            enc_patches = target_patches.clone()
            for i in range(npatches):
                enc_patches[i] = enc_patches[i] * \
                    (1 / (enc_patches[i].norm(2) + 1e-8))

            conv_enc = nn.Conv2d(
                npatches,
                npatches,
                kernel_size=1,
                stride=stride,
                bias=False,
                groups=npatches,
            )
            conv_enc.weight.data = enc_patches
            return conv_enc

            # normalize is not needed, it doesn't change the result!
            if normalize:
                raise NotImplementedError

            if interpolate:
                raise NotImplementedError
        else:

            conv_dec = nn.ConvTranspose2d(
                npatches, C, kernel_size=patch_size, stride=stride, bias=False
            )
            conv_dec.weight.data = target_patches
            return conv_dec

    def _extract_patches(self, img, patch_size, stride):
        n_dim = 3
        assert img.dim() == n_dim, "image must be of dimension 3."
        kH, kW = patch_size, patch_size
        dH, dW = stride, stride
        input_windows = img.unfold(1, kH, dH).unfold(2, kW, dW)
        i_1, i_2, i_3, i_4, i_5 = (
            input_windows.size(0),
            input_windows.size(1),
            input_windows.size(2),
            input_windows.size(3),
            input_windows.size(4),
        )
        input_windows = (
            input_windows.permute(1, 2, 0, 3, 4)
            .contiguous()
            .view(i_2 * i_3, i_1, i_4, i_5)
        )
        patches_all = input_windows
        return patches_all


class BASE(nn.Module):
    def __init__(self, inner_nc, dev: torch.device = None):
        super().__init__()
        se = SELayer(inner_nc, 16)
        model = [se]
        gus = gussin(1.5)
        self.gus = torch.unsqueeze(gus, 1).double()
        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.dev = dev

    def forward(self, x):
        Nonparm = Selfpatch()
        out_32 = self.model(x)
        b, c, h, w = out_32.size()
        gus = self.gus.float()
        gus_out = []
        for i in range(b):
            go = out_32[i].expand(h * w, c, h, w)
            go = gus.to(self.dev) * go
            go = torch.sum(go, -1)
            go = torch.sum(go, -1)
            go = go.contiguous().view(1, c, h, w)
            gus_out.append(go)
        gus_out = torch.cat(gus_out)
        csa2_in_org = torch.sigmoid(out_32)
        csa2_f_org = torch.nn.functional.pad(csa2_in_org, (1, 1, 1, 1))
        csa2_ff_org = torch.nn.functional.pad(out_32, (1, 1, 1, 1))
        out_csa = []
        for i in range(b):
            csa2_fff, csa2_f, csa2_conv = Nonparm.buildAutoencoder(
                csa2_f_org[i], csa2_in_org[i], csa2_ff_org[i], 3, 1
            )
            csa2_conv = csa2_conv.expand_as(csa2_f)
            csa_a = csa2_conv * csa2_f
            csa_a = torch.mean(csa_a, 1)
            a_c, a_h, a_w = csa_a.size()
            csa_a = csa_a.contiguous().view(a_c, -1)
            csa_a = F.softmax(csa_a, dim=1)
            csa_a = csa_a.contiguous().view(a_c, 1, a_h, a_h)
            out = csa_a * csa2_fff
            out = torch.sum(out, -1)
            out = torch.sum(out, -1)
            out = out.contiguous().view(1, c, h, w)
            out_csa.append(out)
        out_csa = torch.cat(out_csa)
        out_32 = torch.cat([gus_out, out_csa], 1)
        out_32 = self.down(out_32)
        return out_32


class UnetSkipConnectionEBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        downconv = nn.Conv2d(
            outer_nc,
            inner_nc,
            kernel_size=4,
            stride=2,
            padding=1)

        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = nn.InstanceNorm2d(inner_nc, affine=True)
        if outermost:
            down = [downconv]
            model = down
        elif innermost:
            down = [downrelu, downconv]
            model = down
        else:
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                model = down + [nn.Dropout(0.5)]
            else:
                model = down
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UnetSkipConnectionDBlock(nn.Module):
    def __init__(
        self,
        inner_nc,
        outer_nc,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        uprelu = nn.ReLU(True)
        upnorm = nn.InstanceNorm2d(outer_nc, affine=True)
        upconv = nn.ConvTranspose2d(
            inner_nc, outer_nc, kernel_size=4, stride=2, padding=1
        )
        up = [uprelu, upconv, upnorm]

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=0,
                dilation=dilation,
                bias=False,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ConvDown(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        kernel,
        stride,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        layers=1,
        activ=True,
    ):
        super().__init__()
        nf_mult = 1
        nums = out_c / 64
        sequence = []

        for i in range(1, layers + 1):
            nf_mult_prev = nf_mult
            if nums == 8:
                if in_c == 512:

                    nf_mult = 1
                else:
                    nf_mult = 2

            else:
                nf_mult = min(2 ** i, 8)
            if kernel != 1:

                if activ and layers == 1:
                    sequence += [
                        nn.Conv2d(
                            nf_mult_prev * in_c,
                            nf_mult * in_c,
                            kernel_size=kernel,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                        nn.InstanceNorm2d(nf_mult * in_c),
                    ]
                else:
                    sequence += [
                        nn.Conv2d(
                            nf_mult_prev * in_c,
                            nf_mult * in_c,
                            kernel_size=kernel,
                            stride=stride,
                            padding=padding,
                            bias=bias,
                        ),
                        nn.InstanceNorm2d(nf_mult * in_c),
                        nn.LeakyReLU(0.2, True),
                    ]

            else:

                sequence += [
                    nn.Conv2d(
                        in_c,
                        out_c,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                        bias=bias,
                    ),
                    nn.InstanceNorm2d(out_c),
                    nn.LeakyReLU(0.2, True),
                ]

            if not activ:
                if i + 1 == layers:
                    if layers == 2:
                        sequence += [
                            nn.Conv2d(
                                nf_mult * in_c,
                                nf_mult * in_c,
                                kernel_size=kernel,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                            ),
                            nn.InstanceNorm2d(nf_mult * in_c),
                        ]
                    else:
                        sequence += [
                            nn.Conv2d(
                                nf_mult_prev * in_c,
                                nf_mult * in_c,
                                kernel_size=kernel,
                                stride=stride,
                                padding=padding,
                                bias=bias,
                            ),
                            nn.InstanceNorm2d(nf_mult * in_c),
                        ]
                    break

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ConvUp(nn.Module):
    def __init__(
            self,
            in_c,
            out_c,
            kernel,
            stride,
            padding=0,
            dilation=1,
            groups=1,
            bias=False):
        super().__init__()

        self.conv = nn.Conv2d(
            in_c, out_c, kernel, stride, padding, dilation, groups, bias
        )
        self.bn = nn.InstanceNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, size):
        out = F.interpolate(input=input, size=size, mode="bilinear")
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out
