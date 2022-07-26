import io
import time

import torch
import cv2
from logzero import logger as log
from PIL import Image
from PIL.Image import Image as PILImage
import numpy as np
import pickle
import skia
from src.skia_lib import skia_paintor as skp
from src.skia_lib import skia_util as sku

def draw_text(canvas, tfd, bg_img, draw_offset_y, draw_offset_x, pivot_y, pivot_x, font, text, text_top, i, angle=0):
    height, width, _, = bg_img.shape
    textblob = skia.TextBlob(text, font)
    shadow_param, fill_param, _, stroke_param = tfd.effect_params[i].get_data()
    # get paints
    fill_paint = skp.get_fill_paint(fill_param)
    shadow_paint = skp.get_shadow_paint(shadow_param)
    stroke_paint = skp.get_stroke_paint(stroke_param)
    fill_alpha = skp.get_fill_alpha(height,width, textblob, float(draw_offset_x), float(draw_offset_y),pivot_x, pivot_y,angle)
    stroke_alpha = skp.get_stroke_alpha(height,width, textblob, float(draw_offset_x), float(draw_offset_y), pivot_x, pivot_y, stroke_param,angle)
    shadow_bitmap, shadow_alpha = skp.get_shadow_bitmap_and_alpha(height,width, shadow_param, fill_alpha, shadow_paint)
    # render effects
    shadow_visibility_flag, fill_visibility_flag, gardation_visibility_flag, stroke_visibility_flag = tfd.effect_visibility[i].get_data()
    if shadow_visibility_flag==True:
        (op,bsz,dp,theta,shift,offsetds_y,offsetds_x,shadow_color) = shadow_param
        canvas = skp.render_bitmap(canvas, shadow_paint, shadow_bitmap, offsetds_x,offsetds_y)
    canvas.rotate(angle, pivot_x, pivot_y)
    if fill_visibility_flag==True:
        canvas = skp.render_fill(canvas, textblob, float(draw_offset_x), float(draw_offset_y), fill_paint)
    if stroke_visibility_flag==True:
        canvas = skp.render_stroke(canvas, textblob, float(draw_offset_x), float(draw_offset_y), stroke_paint)
    canvas.resetMatrix()
    return canvas

def render_tfd(tfd):
    if type(tfd.bg)==str:
        bg_img = cv2.imread(tfd.bg)[:,:,::-1]
    else:
        bg_img = tfd.bg
    height, width, _, = bg_img.shape
    surface, canvas = skp.get_canvas(height,width, img=bg_img)
    for i, text in enumerate(tfd.texts):
        draw_offset_y, draw_offset_x = tfd.text_offsets[i]
        pivot_y, pivot_x = tfd.text_pivots[i]
        draw_offset_x = draw_offset_x / max(float(tfd.text_form_data[i].width_scale),1e-5)
        font_size, font_id, font_path = tfd.font_data[i].get_data()
        text_offset = font_size*0.1
        font = sku.get_textfont(font_id, font_size, font_path=font_path)
        text_spatial_info = sku.get_text_spatial_info(text_offset, text_offset, font, text)
        (text_top,_),(text_height, text_width),bboxes,_ = text_spatial_info
        angle = tfd.text_form_data[i].angle
        canvas.scale(tfd.text_form_data[i].width_scale, 1)
        canvas = draw_text(
            canvas, 
            tfd, 
            bg_img, 
            draw_offset_y, 
            draw_offset_x, 
            pivot_y, 
            pivot_x,
            font, 
            text, 
            text_top, 
            i, 
            angle
        )
        canvas.resetMatrix()
    rendered_img = surface.makeImageSnapshot().toarray()[:,:,0:3][:,:,::-1]
    return rendered_img

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Render an image from training format data.')
    parser.add_argument('--filename', type=str,default='gen_data/load_eng_tmp/metadata/0_0.pkl',help='filename for rendering')
    args,_ = parser.parse_known_args()
    tfd = pickle.load(open(args.filename, 'rb'))
    output_img = render_tfd(tfd)

