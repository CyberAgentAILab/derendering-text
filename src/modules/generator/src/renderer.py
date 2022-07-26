import numpy as np
import skia
from src.dto.dto_generator import TextGeneratorInputHandler
from .data_handler import DataHandler
from src.skia_lib import skia_paintor as skp
from src.skia_lib import skia_util as sku
from .synthtext_lib import synthtext_util as stu
from src.dto.dto_generator import RenderingData

class Renderer(object):
    
    def render(self, ih: TextGeneratorInputHandler, dh: DataHandler):
        # load data for rendering
        current_img = dh.get_canvas_img()
        img_height, img_width = current_img.shape[0:2]
        vertical_text_flag, rotation_text_flag = dh.tmp.get_text_type_flags()
        # get_canvas
        surface, canvas = skp.get_canvas(
            img_height, img_width, img=current_img)
        if ih.use_homography:
            canvas, alpha = self.draw_text_with_stynthtext_rule(
                canvas, ih, dh, img_height, img_width)
        elif vertical_text_flag == 1:
            canvas, alpha = self.draw_vertical_text(
                canvas, ih, dh, img_height, img_width)
        else:
            canvas, alpha = self.draw_horizontal_text(
                canvas, ih, dh, img_height, img_width)

        rendered_img = surface.makeImageSnapshot().toarray()[
            :, :, 0:3][:, :, ::-1]
        dh.set_canvas_img(rendered_img)
        dh.update_alpha(alpha)

    def draw_horizontal_text(
            self,
            canvas: skia.Canvas,
            ih: TextGeneratorInputHandler,
            dh: DataHandler,
            img_height: int,
            img_width: int):
        texts = dh.tmp.get_texts()
        # initialization
        shadow_alpha_all = np.zeros((img_height, img_width), dtype=np.float32)
        fill_alpha_all = np.zeros((img_height, img_width), dtype=np.float32)
        stroke_alpha_all = np.zeros((img_height, img_width), dtype=np.float32)
        dh.box_alpha_mask_init()
        for text_index, text in enumerate(texts):
            # get data for rendering each text
            rd = dh.get_text_rendering_data(text_index)
            # render
            canvas = self.render_text_with_skia(canvas, rd)
            # get alpha of the rendered text
            shadow_alpha, fill_alpha, stroke_alpha = rd.get_alpha()
            # update
            shadow_alpha_all = np.maximum(shadow_alpha, shadow_alpha_all)
            fill_alpha_all = np.maximum(fill_alpha, fill_alpha_all)
            stroke_alpha_all = np.maximum(stroke_alpha, stroke_alpha_all)
            dh.box_alpha_mask_update(rd)
        return canvas, (shadow_alpha_all, fill_alpha_all, stroke_alpha_all)

    def draw_vertical_text(
            self,
            canvas: skia.Canvas,
            ih: TextGeneratorInputHandler,
            dh: DataHandler,
            img_height: int,
            img_width: int):
        texts = dh.tmp.get_texts()
        # initialization
        shadow_alpha_all = np.zeros((img_height, img_width), dtype=np.float32)
        fill_alpha_all = np.zeros((img_height, img_width), dtype=np.float32)
        stroke_alpha_all = np.zeros((img_height, img_width), dtype=np.float32)
        dh.box_alpha_mask_init()
        for text_index, text in enumerate(texts):
            for char_index, char in enumerate(text):
                # get data for rendering each character
                rd = dh.get_char_rendering_data(text_index, char_index)
                # render
                canvas = self.render_text_with_skia(canvas, rd)
                # get alpha of the rendered text
                shadow_alpha, fill_alpha, stroke_alpha = rd.get_alpha()
                # update
                shadow_alpha_all = np.maximum(shadow_alpha, shadow_alpha_all)
                fill_alpha_all = np.maximum(fill_alpha, fill_alpha_all)
                stroke_alpha_all = np.maximum(stroke_alpha, stroke_alpha_all)
                dh.box_alpha_mask_update(rd)
        return canvas, (shadow_alpha_all, fill_alpha_all, stroke_alpha_all)

    def render_text_with_skia(self, canvas: skia.Canvas, rd: RenderingData):
        # rendering effects
        if rd.shadow_visibility_flag:
            canvas = skp.render_bitmap(
                canvas,
                rd.shadow_paint,
                rd.shadow_bitmap,
                rd.shadow_offset_x,
                rd.shadow_offset_y)
        canvas.rotate(rd.angle, rd.offset_x, rd.offset_y)
        if rd.fill_visibility_flag:
            canvas = skp.render_fill(
                canvas,
                rd.textblob,
                rd.offset_x,
                rd.offset_y,
                rd.fill_paint)
        if rd.gardation_visibility_flag:
            canvas = skp.render_gradation(
                canvas, rd.textblob, rd.offset_x, rd.offset_y, rd.grad_paint, rd.grad_blend_mode)
        if rd.stroke_visibility_flag:
            canvas = skp.render_stroke(
                canvas,
                rd.textblob,
                rd.offset_x,
                rd.offset_y,
                rd.stroke_paint)
        canvas.resetMatrix()
        return canvas

    def transform_bboxes(
            self,
            bboxes: np.ndarray,
            hminv: np.ndarray) -> np.ndarray:
        bboxes = stu.yx2xy(bboxes)
        bboxes = stu.homographyBB(bboxes, hminv)
        bboxes = stu.xy2yx(bboxes)
        return bboxes

    def draw_text_with_stynthtext_rule(
            self,
            canvas: skia.Canvas,
            ih: TextGeneratorInputHandler,
            dh: DataHandler,
            img_height: int,
            img_width: int):
        texts = dh.tmp.get_texts()
        img_height, img_width = ih.get_img_size()
        hm, hminv = ih.get_homography()
        bboxes = dh.get_bboxes(-1)
        bboxes = self.transform_bboxes(bboxes, hminv)
        dh.set_bboxes(bboxes, -1)
        # initialization
        shadow_alpha_all = np.zeros((img_height, img_width), dtype=np.float32)
        fill_alpha_all = np.zeros((img_height, img_width), dtype=np.float32)
        stroke_alpha_all = np.zeros((img_height, img_width), dtype=np.float32)
        dh.box_alpha_mask_init()
        for text_index, text in enumerate(texts):
            rd = dh.get_text_rendering_data(text_index)
            # rendering
            canvas, (shadow_alpha, fill_alpha, stroke_alpha) = self.render_text_with_synthtext_rule(
                canvas, rd, img_height, img_width, hm)
            # update
            shadow_alpha_all = np.maximum(shadow_alpha, shadow_alpha_all)
            fill_alpha_all = np.maximum(fill_alpha, fill_alpha_all)
            stroke_alpha_all = np.maximum(stroke_alpha, stroke_alpha_all)
            dh.box_alpha_mask_update(rd)
        return canvas, (shadow_alpha_all, fill_alpha_all, stroke_alpha_all)

    def render_text_with_synthtext_rule(
            self,
            canvas: skia.Canvas,
            rd: RenderingData,
            img_height: int,
            img_width: int,
            hm: np.ndarray,
            op: float = 1.0):
        fill_alpha = stu.warpHomography(
            rd.fill_alpha, hm, (img_width, img_height))
        fill_bitmap = skp.alpha2bitmap(
            img_height, img_width, fill_alpha, op=op)
        stroke_alpha = stu.warpHomography(
            rd.stroke_alpha, hm, (img_width, img_height))
        stroke_bitmap = skp.alpha2bitmap(img_height, img_width, stroke_alpha)
        shadow_alpha = stu.warpHomography(
            rd.shadow_alpha, hm, (img_width, img_height))
        shadow_bitmap = skp.alpha2bitmap(img_height, img_width, shadow_alpha)
        if rd.shadow_visibility_flag:
            canvas = skp.render_bitmap(
                canvas,
                rd.shadow_paint,
                shadow_bitmap,
                rd.shadow_offset_x,
                rd.shadow_offset_y)
        if rd.fill_visibility_flag:
            canvas = skp.render_bitmap(canvas, rd.fill_paint, fill_bitmap)
        if rd.stroke_visibility_flag:
            canvas = skp.render_bitmap(canvas, rd.stroke_paint, stroke_bitmap)
        return canvas, (shadow_alpha, fill_alpha, stroke_alpha)
