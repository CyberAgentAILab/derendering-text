import numpy as np
from typing import Tuple
from src.dto.dto_postprocess import VectorData
from src.skia_lib import skia_paintor as skp
from src.skia_lib import skia_util as sku


def get_offset(i, vd) -> Tuple[int, int]:
    x0, _, _, _ = vd.tb_param[i].box
    offset_x = int(
        x0 / max(float(vd.tb_param[i].text_form_data.width_scale), 1e-5))
    offset_y = int(vd.tb_param[i].y_start - vd.tb_param[i].text_top)
    return (offset_y, offset_x)


def render_vd(vd: VectorData) -> np.ndarray:
    """Perform actual rendering routine"""
    height, width = vd.bg.shape[0:2]
    surface, canvas = skp.get_canvas(height, width, img=vd.bg.copy())
    for i in range(len(vd.tb_param)):
        # load font data
        font_size, font_id, font_path = vd.tb_param[i].font_data.get_data()
        font = sku.get_textfont(font_id, font_size)
        textblob = sku.get_textblob(vd.tb_param[i].text, font)
        # get offsets
        offsets = get_offset(i, vd)
        offset_y, offset_x = offsets
        # load effect params
        effect_params = vd.effect_param[i].get_data()
        paints = skp.get_paint(effect_params)
        shadow_paint, fill_paint, stroke_paint, grad_paint = paints
        # load effect visibility
        shadow_visibility_flag, fill_visibility_flag, gardation_visibility_flag, stroke_visibility_flag = vd.effect_visibility[i].get_data(
        )
        # get alpha maps
        shadow_alpha, fill_alpha, stroke_alpha, shadow_bitmap = skp.get_alpha(
            (height, width), textblob, offsets, effect_params, paints, angle=0)
        # draw texts
        canvas.resetMatrix()
        canvas.scale(vd.tb_param[i].text_form_data.width_scale, 1)
        if shadow_visibility_flag:
            canvas = skp.render_bitmap(
                canvas,
                shadow_paint,
                shadow_bitmap,
                vd.effect_param[i].shadow_param.offset_x,
                vd.effect_param[i].shadow_param.offset_y)
        canvas = skp.render_fill(
            canvas,
            textblob,
            float(offset_x),
            float(offset_y),
            fill_paint)
        if stroke_visibility_flag:
            canvas = skp.render_stroke(
                canvas,
                textblob,
                float(offset_x),
                float(offset_y),
                stroke_paint)
    return surface.makeImageSnapshot().toarray()[:, :, 0:3][:, :, ::-1]
