import torch
import torch.nn as nn

from .layers.renderer import (
    AffineTransformer,
    AlphaRenderer,
    ShadowAlphaTransformer,
    StrokeAlphaRenderer,
    char_mask_pooling,
    compositer,
    get_global_alpha,
    get_max_char_box_num,
)


class Reconstructor(nn.Module):
    def __init__(self, dev: torch.device = None):
        super().__init__()
        self.alpha_renderer = AlphaRenderer(dev=dev)
        self.stroke_alpha_renderer = StrokeAlphaRenderer(dev=dev)
        self.affine_transformer_paraminput = AffineTransformer()
        self.shadow_tranformer = ShadowAlphaTransformer(dev=dev)
        self.dev = dev

    def forward(self, features_pix, img_org, bg_img, ti, colors=None):
        ##################
        # data preparing
        ##################
        # variables
        recognition_results = ti.ocr_outs[2]
        # effect information
        (
            shadow_param_sig_outs,
            shadow_param_tanh_outs,
            stroke_param_outs,
        ) = ti.effect_param_outs
        # bbox_information
        text_instance_mask = ti.bbox_information.get_text_instance_mask()
        text_rectangles = ti.bbox_information.get_text_rectangle()
        char_instance_mask = ti.bbox_information.get_char_instance_mask()
        char_rectangles = ti.bbox_information.get_char_rectangle()
        char_labels = ti.bbox_information.get_char_label()
        char_sizes = ti.bbox_information.get_char_size()
        charindex2textindex = ti.bbox_information.get_charindex2textindex()
        # mask shape
        text_instance_mask = torch.from_numpy(text_instance_mask).to(self.dev)
        text_instance_mask = text_instance_mask.unsqueeze(1).float()
        char_instance_mask = torch.from_numpy(char_instance_mask).to(self.dev)
        char_instance_mask = char_instance_mask.unsqueeze(1).float()

        # pool character information
        # convert character prediction map to character vector
        # recognition_results [N x C x H x W]
        # -> char_rec_vec [N x B x C]
        # B is number of bounding box
        char_rec_vec = char_mask_pooling(
            recognition_results,
            char_rectangles,
            char_instance_mask,
            dev=self.dev)

        ######################
        # processings
        # - convert text condition to text alpha
        # - transform text alpha by affine transformation
        # - locate text alpha to raster space
        # - composite image
        #######################

        ######################
        # convert text condition to text alpha
        #######################
        # rendering alpha
        # text condition to text alpha
        # prerendered_fill_alpha List:[1 x 1x 64 x 64]xB
        prerendered_fill_alpha = self.alpha_renderer(
            ti.font_outs, char_labels, char_rec_vec, charindex2textindex
        )  #
        # rendering stroke alpha
        # prerendered_fill_alpha List:[1 x 1x 64 x 64]xB
        prerendered_stroke_alpha = self.stroke_alpha_renderer(
            ti.font_outs,
            stroke_param_outs,
            char_labels,
            char_rec_vec,
            charindex2textindex,
        )  #

        ######################
        # transform text alpha by affine transformation
        #######################
        max_char_box_num = get_max_char_box_num(char_rectangles)
        affine_outs = (
            torch.zeros(features_pix.shape[0] * max_char_box_num, 6, 1, 1)
            .float()
            .to(self.dev)
        )
        (
            prerendered_fill_alpha_affine,
            prerendered_stroke_alpha_affine,
        ) = self.affine_transformer_paraminput(
            affine_outs,
            prerendered_fill_alpha,
            prerendered_stroke_alpha,
            char_sizes,
        )  #
        ######################
        # locate text alpha to raster space
        #######################
        # prerendered_fill_alpha_global is text alpha: [N, 1, H, W]
        # fill_alpha_loc is text instance masks: [N, 1, H, W]
        prerendered_fill_alpha_global, fill_alpha_loc = get_global_alpha(
            prerendered_fill_alpha_affine,
            char_rectangles,
            charindex2textindex,
            img_org.shape[2],  # height
            img_org.shape[3],  # width
            dev=self.dev
        )  #
        # prerendered_stroke_alpha_global is text alpha: [N, 1, H, W]
        # stroke_alpha_loc is text instance masks: [N, 1, H, W]
        prerendered_stroke_alpha_global, stroke_alpha_loc = get_global_alpha(
            prerendered_stroke_alpha_affine,
            char_rectangles,
            charindex2textindex,
            img_org.shape[2],  # height
            img_org.shape[3],  # width
            dev=self.dev
        )  #
        # prerendered_shadow_alpha_global is text alpha: [N, 1, H, W]
        # shadow_alpha_loc is text instance masks: [N, 1, H, W]
        # convert fill alpha to drop shadow by differential functions blur and
        # offsets.
        prerendered_shadow_alpha_global, shadow_alpha_loc = self.shadow_tranformer(
            prerendered_fill_alpha_global,
            ti.font_size_outs,
            shadow_param_sig_outs,
            shadow_param_tanh_outs,
            text_rectangles,
        )
        rendered_alpha_outs = (
            prerendered_fill_alpha_global,
            prerendered_stroke_alpha_global,
            prerendered_shadow_alpha_global,
            fill_alpha_loc,
            stroke_alpha_loc,
            shadow_alpha_loc,
        )
        ######################
        # composite image
        #######################
        # compositing ( get image )
        color_maps, rgb_reconstructed, colors_pred = compositer(
            ti.alpha_outs,
            rendered_alpha_outs,
            ti.effect_visibility_outs,
            img_org,
            bg_img,
            colors,
            text_instance_mask,
            dev=self.dev
        )
        return (
            rendered_alpha_outs,
            color_maps,
            rgb_reconstructed,
            affine_outs,
            char_rec_vec,
            colors_pred,
        )

    def reconstruction_with_vector_elements(self, optp, fix_params):
        # variables
        img_org, bg_img, colors, bbox_information = fix_params
        bg_img = bg_img.detach()
        colors = (optp.fill_color, optp.shadow_color, optp.stroke_color)
        text_instance_mask = bbox_information.get_text_instance_mask()
        text_rectangles = bbox_information.get_text_rectangle()
        char_rectangles = bbox_information.get_char_rectangle()
        char_labels = bbox_information.get_char_label()
        char_sizes = bbox_information.get_char_size()
        charindex2textindex = bbox_information.get_charindex2textindex()
        if img_org.is_cuda:
            text_instance_mask = torch.from_numpy(
                text_instance_mask).to(self.dev)
        else:
            text_instance_mask = torch.from_numpy(text_instance_mask)
        text_instance_mask = text_instance_mask.unsqueeze(1).float()
        # rendering alpha
        prerendered_fill_alpha = self.alpha_renderer(
            optp.font_outs, char_labels, optp.char_vec, charindex2textindex
        )
        # rendering stroke alpha
        prerendered_stroke_alpha = self.stroke_alpha_renderer(
            optp.font_outs,
            optp.stroke_param_outs,
            char_labels,
            optp.char_vec,
            charindex2textindex,
        )

        # affine transformation
        (
            prerendered_fill_alpha_affine,
            prerendered_stroke_alpha_affine,
        ) = self.affine_transformer_paraminput(
            optp.affine_outs,
            prerendered_fill_alpha,
            prerendered_stroke_alpha,
            char_sizes,
        )  #
        # get global alpha
        prerendered_fill_alpha_global, fill_alpha_loc = get_global_alpha(
            prerendered_fill_alpha_affine,
            char_rectangles,
            charindex2textindex,
            img_org.shape[2],  # height
            img_org.shape[3],  # width
            dev=self.dev
        )  #
        prerendered_stroke_alpha_global, stroke_alpha_loc = get_global_alpha(
            prerendered_stroke_alpha_affine,
            char_rectangles,
            charindex2textindex,
            img_org.shape[2],  # height
            img_org.shape[3],  # width
            dev=self.dev
        )  #
        # get shadow alpha
        prerendered_shadow_alpha_global, shadow_alpha_loc = self.shadow_tranformer(
            prerendered_fill_alpha_global,
            None,
            optp.shadow_param_sig_outs,
            optp.shadow_param_tanh_outs,
            text_rectangles,
        )
        rendered_alpha_outs = (
            prerendered_fill_alpha_global,
            prerendered_stroke_alpha_global,
            prerendered_shadow_alpha_global,
            fill_alpha_loc,
            stroke_alpha_loc,
            shadow_alpha_loc,
        )
        # compositing
        effect_visibility_outs = (
            optp.shadow_visibility_outs,
            optp.stroke_visibility_outs,
        )
        _, rgb_reconstructed, _ = compositer(
            optp.alpha_outs,
            rendered_alpha_outs,
            effect_visibility_outs,
            img_org,
            bg_img,
            colors,
            text_instance_mask,
            dev=self.dev
        )
        return rendered_alpha_outs, rgb_reconstructed
