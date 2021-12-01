
import torch
from torch import nn
from torch.functional import F
from torch.autograd import Variable
import skimage.measure
import time
import numpy as np
import math


class TextParserLossFunc(nn.Module):
    def __init__(self, text_pool_num=10):
        super().__init__()
        self.text_pool_num = text_pool_num
        self.label_num = 5
        self.stroke_param_num = 1
        self.oglow_param_num = 5
        self.ignore_labl = -255

    def compute_geo_loss(self, geomap_pred, geomap_gt, scoremap_gt):
        if torch.sum(scoremap_gt == 1).item() == 0:
            return torch.FloatTensor([0]).cuda()
        else:
            # t, b, l, r
            d1_pred, d2_pred, d3_pred, d4_pred = torch.split(geomap_pred, 1, 1)
            d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(geomap_gt, 1, 1)
            area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
            area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
            w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
            h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
            area_intersect = w_union * h_union
            area_union = area_gt + area_pred - area_intersect
            geo_loss = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
            geo_loss = geo_loss[(scoremap_gt == 1)]
            geo_loss = torch.mean(geo_loss)
            return geo_loss

    def compute_angle_loss(self, theta_pred, geomap_gt, scoremap_gt):
        if torch.sum(scoremap_gt > 0).item() == 0:
            return torch.FloatTensor([0]).cuda()
        else:
            # t, b, l, r
            d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(geomap_gt, 1, 1)
            angle_loss = 1 - torch.cos(theta_pred - theta_gt)
            angle_loss = torch.mean(angle_loss[scoremap_gt > 0])
            return angle_loss

    def alpha_estimation_loss(self, alpha_pred, alpha_gt, text_ins_mask):
        text_ins_mask = text_ins_mask.unsqueeze(1).float()
        fill_alpha_fg_loss = torch.FloatTensor([0]).cuda()
        fill_alpha_bg_loss = torch.FloatTensor([0]).cuda()
        shadow_alpha_fg_loss = torch.FloatTensor([0]).cuda()
        shadow_alpha_bg_loss = torch.FloatTensor([0]).cuda()
        stroke_alpha_fg_loss = torch.FloatTensor([0]).cuda()
        stroke_alpha_bg_loss = torch.FloatTensor([0]).cuda()
        fill_alpha_fg_cnt = 0
        fill_alpha_bg_cnt = 0
        shadow_alpha_fg_cnt = 0
        shadow_alpha_bg_cnt = 0
        stroke_alpha_fg_cnt = 0
        stroke_alpha_bg_cnt = 0
        for i in range(len(alpha_pred)):
            knum = int(torch.max(text_ins_mask[i]).item())
            for k in range(knum):
                alpha_index = 0
                instance_loc = text_ins_mask[i:i + 1] == (k + 1)
                loc = instance_loc & (
                    alpha_gt[i:i + 1, alpha_index:alpha_index + 1] > 0)
                if torch.sum(loc == 1).item() > 0:
                    fill_alpha_fg_loss += F.mse_loss(alpha_pred[i:i +
                                                                1, alpha_index:alpha_index +
                                                                1][loc], alpha_gt[i:i +
                                                                                  1, alpha_index:alpha_index +
                                                                                  1][loc])
                    fill_alpha_fg_cnt += 1
                loc = instance_loc & (
                    alpha_gt[i:i + 1, alpha_index:alpha_index + 1] == 0)
                if torch.sum(loc == 1).item() > 0:
                    fill_alpha_bg_loss += F.mse_loss(alpha_pred[i:i +
                                                                1, alpha_index:alpha_index +
                                                                1][loc], alpha_gt[i:i +
                                                                                  1, alpha_index:alpha_index +
                                                                                  1][loc])
                    fill_alpha_bg_cnt += 1

                alpha_index = 1
                loc = instance_loc & (
                    alpha_gt[i:i + 1, alpha_index:alpha_index + 1] > 0)
                if torch.sum(loc == 1).item() > 0:
                    shadow_alpha_fg_loss += F.mse_loss(alpha_pred[i:i +
                                                                  1, alpha_index:alpha_index +
                                                                  1][loc], alpha_gt[i:i +
                                                                                    1, alpha_index:alpha_index +
                                                                                    1][loc])
                    shadow_alpha_fg_cnt += 1
                loc = instance_loc & (
                    alpha_gt[i:i + 1, alpha_index:alpha_index + 1] == 0)
                if torch.sum(loc == 1).item() > 0:
                    shadow_alpha_bg_loss += F.mse_loss(alpha_pred[i:i +
                                                                  1, alpha_index:alpha_index +
                                                                  1][loc], alpha_gt[i:i +
                                                                                    1, alpha_index:alpha_index +
                                                                                    1][loc])
                    shadow_alpha_bg_cnt += 1

                alpha_index = 2
                loc = instance_loc & (
                    alpha_gt[i:i + 1, alpha_index:alpha_index + 1] > 0)
                if torch.sum(loc == 1).item() > 0:
                    stroke_alpha_fg_loss += F.mse_loss(alpha_pred[i:i +
                                                                  1, alpha_index:alpha_index +
                                                                  1][loc], alpha_gt[i:i +
                                                                                    1, alpha_index:alpha_index +
                                                                                    1][loc])
                    stroke_alpha_fg_cnt += 1
                loc = instance_loc & (
                    alpha_gt[i:i + 1, alpha_index:alpha_index + 1] == 0)
                if torch.sum(loc == 1).item() > 0:
                    stroke_alpha_bg_loss += F.mse_loss(alpha_pred[i:i +
                                                                  1, alpha_index:alpha_index +
                                                                  1][loc], alpha_gt[i:i +
                                                                                    1, alpha_index:alpha_index +
                                                                                    1][loc])
                    stroke_alpha_bg_cnt += 1

        fill_alpha_fg_loss /= max(fill_alpha_fg_cnt, 1)
        fill_alpha_bg_loss /= max(fill_alpha_bg_cnt, 1)
        shadow_alpha_fg_loss /= max(shadow_alpha_fg_cnt, 1)
        shadow_alpha_bg_loss /= max(shadow_alpha_bg_cnt, 1)
        stroke_alpha_fg_loss /= max(stroke_alpha_fg_cnt, 1)
        stroke_alpha_bg_loss /= max(stroke_alpha_bg_cnt, 1)

        return fill_alpha_fg_loss, fill_alpha_bg_loss, \
            shadow_alpha_fg_loss, shadow_alpha_bg_loss, stroke_alpha_fg_loss, stroke_alpha_bg_loss

    def effect_visibility_loss(
            self,
            effect_visibility_outs,
            shadow_data,
            stroke_data,
            valid_indexes):
        shadow_visibility, stroke_visibility = effect_visibility_outs
        shadow_visibility = shadow_visibility.view(
            shadow_visibility.shape[0],
            shadow_visibility.shape[1],
            shadow_visibility.shape[2])
        stroke_visibility = stroke_visibility.view(
            stroke_visibility.shape[0],
            stroke_visibility.shape[1],
            stroke_visibility.shape[2])
        (_, shadow_visibility_gt) = shadow_data
        (_, stroke_visibility_gt) = stroke_data
        # label
        shadow_visibility_loss = torch.FloatTensor([0]).cuda()
        stroke_visibility_loss = torch.FloatTensor([0]).cuda()
        gcnt, scnt = 0, 0
        text_num = self.text_pool_num
        for i in range(len(shadow_visibility_gt)):
            for k in range(text_num):
                if valid_indexes[i, k].item() == 0:
                    continue
                if shadow_visibility_gt[i, k:k + 1] != self.ignore_labl:
                    shadow_visibility_loss += F.cross_entropy(
                        shadow_visibility[i, k:k + 1], shadow_visibility_gt[i, k:k + 1].long(), ignore_index=self.ignore_labl)
                    gcnt += 1
                if stroke_visibility_gt[i, k:k + 1] != self.ignore_labl:
                    stroke_visibility_loss += F.cross_entropy(
                        stroke_visibility[i, k:k + 1], stroke_visibility_gt[i, k:k + 1].long(), ignore_index=self.ignore_labl)
                    scnt += 1
        shadow_visibility_loss /= max(gcnt, 1)
        stroke_visibility_loss /= max(scnt, 1)
        return shadow_visibility_loss, stroke_visibility_loss

    def effect_param_loss(
            self,
            effect_param_outs,
            shadow_data,
            stroke_data,
            valid_indexes):
        valid_indexes = valid_indexes.view(
            valid_indexes.shape[0] * valid_indexes.shape[1])
        shadow_param_sig, shadow_param_tanh, stroke_param = effect_param_outs
        (shadow_param_gt, _) = shadow_data
        (stroke_param_gt, _) = stroke_data
        shadow_param_sig = shadow_param_sig.view(
            valid_indexes.shape[0],
            shadow_param_sig.shape[2])[
            valid_indexes == 1]
        shadow_param_tanh = shadow_param_tanh.view(
            valid_indexes.shape[0],
            shadow_param_tanh.shape[2])[
            valid_indexes == 1]
        shadow_param_gt = shadow_param_gt.view(
            valid_indexes.shape[0],
            shadow_param_gt.shape[2])[
            valid_indexes == 1]
        if torch.sum(shadow_param_gt != self.ignore_labl).item() == 0:
            shadow_param_loss = torch.FloatTensor([0]).cuda()
        else:
            shadow_param_sig_loss = F.mse_loss(shadow_param_sig[shadow_param_gt[:, 0:2] != self.ignore_labl],
                                               shadow_param_gt[:, 0:2][shadow_param_gt[:, 0:2] != self.ignore_labl])
            shadow_param_tanh_loss = F.mse_loss(shadow_param_tanh[shadow_param_gt[:, 3:5] != self.ignore_labl],
                                                shadow_param_gt[:, 3:5][shadow_param_gt[:, 3:5] != self.ignore_labl])
            shadow_param_loss = shadow_param_sig_loss + shadow_param_tanh_loss
        if torch.sum(stroke_param_gt != self.ignore_labl).item() == 0:
            stroke_param_loss = torch.FloatTensor([0]).cuda()
        else:
            num = stroke_param_gt.shape[0] * stroke_param_gt.shape[1]
            stroke_param_gt = stroke_param_gt.view(num, 1, 1).long()
            stroke_param = stroke_param.view(num, stroke_param.shape[2], 1, 1)
            stroke_param_loss = F.cross_entropy(stroke_param[valid_indexes == 1],
                                                stroke_param_gt[valid_indexes == 1], ignore_index=self.ignore_labl)
        return shadow_param_loss, stroke_param_loss

    def font_loss(self, font_pred, font_label, valid_indexes):
        if torch.sum(font_label != self.ignore_labl).item(
        ) == 0 or torch.sum(valid_indexes).item() == 0:
            font_loss = torch.FloatTensor([0]).cuda()
        else:
            num = font_pred.shape[0] * font_pred.shape[1]
            font_label = font_label.view(
                font_label.shape[0] * font_label.shape[1], 1, 1).long()
            font_pred = font_pred.view(
                font_pred.shape[0] *
                font_pred.shape[1],
                font_pred.shape[2],
                1,
                1)
            valid_indexes = valid_indexes.view(
                valid_indexes.shape[0] * valid_indexes.shape[1])
            font_loss = F.cross_entropy(font_pred[valid_indexes == 1],
                                        font_label[valid_indexes == 1], ignore_index=self.ignore_labl)
        return font_loss

    def modify_char_fg_mask(self, char_fg_mask, char_scoremap):
        char_fg_mask_v = torch.zeros_like(char_fg_mask) + self.ignore_labl
        char_fg_mask_v[char_scoremap == 1] = 1
        char_fg_mask_v[char_scoremap == self.ignore_labl] = self.ignore_labl
        char_fg_mask_v[(char_fg_mask == 1) & (char_scoremap == 0)] = 0
        return char_fg_mask_v

    def modify_text_fg_mask(self, text_fg_mask, text_scoremap):
        text_fg_mask_v = torch.zeros_like(text_fg_mask)
        text_fg_mask_v[text_scoremap == 1] = 1
        text_fg_mask_v[(text_fg_mask == 1) & (
            text_scoremap == 0)] = self.ignore_labl
        return text_fg_mask_v

    def compute_ocr_loss(self, ocr_pred, ocr_data):
        # ocr_pred
        text_preds, char_preds, recognition_results = ocr_pred
        text_fg_pred, text_tblr_pred, text_orient_pred = text_preds
        char_fg_pred, char_tblr_pred, char_orient_pred = char_preds
        # ocr_data
        text_level_data, char_level_data = ocr_data
        text_fg_mask, text_ins_mask, text_scoremap, text_scoremap_qt, text_geomap, text_rectangles_array = text_level_data
        char_fg_mask, char_ins_mask, char_scoremap, char_scoremap_qt, char_geomap, char_cls_mask, char_rectangles_array = char_level_data
        # convert data
        text_fg_pred_resize = F.interpolate(
            text_fg_pred,
            (text_fg_pred.shape[2] * 4,
             text_fg_pred.shape[3] * 4),
            mode='bilinear')
        char_fg_pred = F.interpolate(
            char_fg_pred,
            (char_fg_pred.shape[2] * 4,
             char_fg_pred.shape[3] * 4),
            mode='bilinear')
        char_fg_mask_v = self.modify_char_fg_mask(char_fg_mask, char_scoremap)
        char_fg_loss = F.cross_entropy(
            char_fg_pred,
            char_fg_mask_v.cuda().long(),
            ignore_index=self.ignore_labl)
        # segmentation branch
        text_fg_loss = F.cross_entropy(
            text_fg_pred_resize,
            text_fg_mask.cuda().long(),
            ignore_index=self.ignore_labl)
        char_cls_mask = F.interpolate(char_cls_mask.float().unsqueeze(
            1), recognition_results.shape[2:4]).squeeze(1)
        char_cls_loss = F.cross_entropy(
            recognition_results,
            char_cls_mask.cuda().long(),
            ignore_index=-1)
        # compute loss
        text_geo_loss = self.compute_geo_loss(
            text_tblr_pred,
            text_geomap.cuda().float(),
            text_scoremap_qt.cuda().float().unsqueeze(1))
        text_angle_loss = self.compute_angle_loss(
            text_orient_pred,
            text_geomap.cuda().float(),
            text_scoremap_qt.cuda().float().unsqueeze(1))
        char_scoremap_qt = char_scoremap_qt.cuda().float().unsqueeze(1)
        char_geomap = char_geomap.cuda().float()
        char_geo_loss = self.compute_geo_loss(char_tblr_pred,
                                              char_geomap,
                                              char_scoremap_qt)
        char_angle_loss = self.compute_angle_loss(
            char_orient_pred, char_geomap, char_scoremap_qt)

        detection_branch_loss = text_geo_loss + 10 * text_angle_loss + text_fg_loss
        char_branch_loss = 10 * char_angle_loss + \
            char_cls_loss + char_geo_loss + char_fg_loss
        ocr_loss = detection_branch_loss + char_branch_loss
        ocr_loss_items = (ocr_loss,
                          text_fg_loss, char_fg_loss, char_cls_loss,
                          text_geo_loss, 10 * text_angle_loss,
                          char_geo_loss, 10 * char_angle_loss
                          )
        return ocr_loss, ocr_loss_items

    def get_text_ins_mask(self, ocr_data):
        # ocr_data
        text_level_data, char_level_data = ocr_data
        text_fg_mask, text_ins_mask, text_scoremap, text_scoremap_qt, text_geomap, text_rectangles_array = text_level_data
        return text_ins_mask

    def forward(self, inputs, outputs):
        # inputs
        img_norm, img_org, ocr_data, style_data = inputs
        (alpha, shadow_data, stroke_data, font_label, valid_text_index) = style_data
        # outputs
        textinfo, _ = outputs
        # ocr loss
        ocr_loss, ocr_loss_items = self.compute_ocr_loss(
            textinfo.ocr_outs, ocr_data)
        # style loss
        font_loss = self.font_loss(
            textinfo.font_outs,
            font_label,
            valid_text_index)
        (shadow_visibility_loss, stroke_visibility_loss) = self.effect_visibility_loss(
            textinfo.effect_visibility_outs, shadow_data, stroke_data, valid_text_index)
        (shadow_param_loss, stroke_param_loss) = self.effect_param_loss(
            textinfo.effect_param_outs, shadow_data, stroke_data, valid_text_index)
        alpha_losses = self.alpha_estimation_loss(
            textinfo.alpha_outs, alpha, self.get_text_ins_mask(ocr_data))
        fill_alpha_fg_loss, fill_alpha_bg_loss, \
            shadow_alpha_fg_loss, shadow_alpha_bg_loss, stroke_alpha_fg_loss, stroke_alpha_bg_loss = alpha_losses
        alpha_loss = 10 * fill_alpha_fg_loss + 1 * fill_alpha_bg_loss + 10 * shadow_alpha_fg_loss + \
            1 * shadow_alpha_bg_loss + 10 * stroke_alpha_fg_loss + 1 * stroke_alpha_bg_loss

        visibility_loss = shadow_visibility_loss + stroke_visibility_loss
        param_loss = shadow_param_loss + stroke_param_loss * 0.1
        total_loss = ocr_loss + font_loss * 0.1 + \
            alpha_loss + visibility_loss + param_loss

        shadow_loss = (shadow_param_loss, shadow_visibility_loss)
        stroke_loss = (stroke_param_loss, stroke_visibility_loss)
        effect_loss_items = (shadow_loss, stroke_loss)
        loss_items = (
            ocr_loss_items,
            effect_loss_items,
            alpha_losses,
            font_loss)
        return total_loss, loss_items
