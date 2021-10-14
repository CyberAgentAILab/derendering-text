import os
import skia
import numpy as np
from functools import lru_cache
from src.io import load_font_dicts
from typing import Tuple, List

fontmgr = skia.FontMgr()

def get_font_path(font_id:int):
    font_dict = load_font_dicts()
    font_path = font_dict[font_id]
    return font_path

def load_font_by_skia_format(font_size:int,font_path:str):
    ft = fontmgr.makeFromFile(font_path,0)
    font = skia.Font(ft, font_size, 1.0, 0.0)
    return font

def get_text_blob(font_size:int, font_path:str, text:str):
    font = load_font_by_skia_format(font_size,font_path)
    textblob = skia.TextBlob(texts, font)
    return textblob

def get_bboxes(offset_y:int, offset_x:int, text_top:int, positions:List, rects:List):
    boxes=[]
    for i in range(len(positions)):
        pos = positions[i]
        rect = rects[i]
        top = pos.y()+rect.top()-1*text_top + offset_y
        left = pos.x()+rect.left() + offset_x
        h = rect.height()
        w = rect.width()
        box = (top,left,h,w)
        boxes.append(box)
    return np.array(boxes)

def get_char_offsets_x(offset_x:int, positions:List, rects:List):
    char_offsets_x=[]
    for i in range(len(positions)):
        pos = positions[i]
        rect = rects[i]
        char_offsets_x.append(pos.x()+ offset_x)
    return char_offsets_x

def get_text_coords(offset_y:int, offset_x:int, positions:List, rects:List):
    text_top=0
    text_bottom=0
    for i in range(len(rects)):
        rect = rects[i]
        top = positions[i].y()+rect.top()
        if top<text_top:
            text_top=top
        bottom = positions[i].y()+rect.bottom()
        if bottom>text_bottom:
            text_bottom=bottom
    text_left = positions[0].x()+rects[0].left()
    text_right = positions[-1].x()+rects[-1].right()
    text_height = text_bottom-text_top
    text_width = text_right-text_left
    text_left_pos = positions[0].x()
    return (text_top, text_left_pos),(text_height, text_width)

def get_text_spatial_info(offset_y:int, offset_x:int, font:skia.Font, text:str):
    glyphs = font.textToGlyphs(text)
    positions = font.getPos(glyphs)
    rects = font.getBounds(glyphs)
    (text_top, text_left),(text_height, text_width) = get_text_coords(offset_y, offset_x, positions, rects)
    bboxes = get_bboxes(offset_y, offset_x, text_top, positions, rects)
    char_offsets_x = get_char_offsets_x(offset_x, positions, rects)
    text_top = -1*text_top + offset_y
    text_left += offset_x
    return (text_top, text_left),(text_height, text_width), bboxes, char_offsets_x

def get_textfont(font_id:int, font_size:int, font_path=None, font_dir:str="data_generator/data/fonts/gfonts", font_file_name:str="font_list/latin_ofl_100.txt"):
    if font_path is None:
        #font_dict = load_font_dicts(font_dir=font_path, font_file_name=font_file_name)
        font_dict = load_font_dicts()
        font_path = font_dict[font_id]
    type_face = fontmgr.makeFromFile(font_path, 0)
    font = skia.Font(type_face, font_size, 1.0, 0.0)
    return font

def get_textblob(text:str, font:skia.Font):
    return skia.TextBlob(text, font)


def bb_yxhw2coords(bboxes:np.ndarray):
    n,_ = bboxes.shape
    coords = np.zeros((2,4,n))
    for i in range(n):
        coords[:,:,i] = bboxes[i,:2][:,None] # initialization
        coords[0,1,i] += bboxes[i,2] # + height
        coords[:,2,i] += bboxes[i,2:4] # + height, + width
        coords[1,3,i] += bboxes[i,3] # + width
    return coords

def add_offset_coords(coords:List, offset:Tuple[int]):
    offset_y, offset_x = offset
    new_coords = []
    for (y,x) in coords:
        new_coord = (y+offset_y,x+offset_x)
        new_coords.append(new_coord)
    return new_coords

def add_offset_bboxes(bboxes:np.ndarray, offset:Tuple[int]):
    offset_y, offset_x = offset
    bboxes[0,:,:] += offset_y
    bboxes[1,:,:] += offset_x
    return bboxes
