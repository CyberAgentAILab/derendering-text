from functools import lru_cache
from typing import Tuple
import pickle

import numpy as np
import torch
from PIL import Image

import torchvision.io as tio
import torchvision.transforms as T
from torchvision.io.image import ImageReadMode

from util.path_list import get_char_dict_file_path, get_google_font_path, get_google_font_list_filename

def transform_inputs(
    filepath: str, size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = tio.read_image(filepath, ImageReadMode.RGB)
    img_inp_norm, img_inp_orig = process_model_inputs(image, size)
    pil_img = Image.fromarray(image.permute(1, 2, 0).numpy())
    return img_inp_norm, img_inp_orig, pil_img


def process_model_inputs(
    image: torch.Tensor, size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # for inpaint
    ratio = size / min(image.size()[1],image.size()[2])
    new_ysize = max(int(((ratio * image.size()[1]) // 128) * 128),128)
    new_xsize = max(int(((ratio * image.size()[2]) // 128) * 128),128)
    new_size = (new_ysize, new_xsize)
    normalize = T.Compose(
        [
            T.Resize(new_size),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_inp_norm = normalize(image).unsqueeze(0)
    to_float = T.Compose(
        [
            T.Resize(new_size),
            T.ConvertImageDtype(torch.float),
            T.Lambda(lambda x: (x - 0.5) * 2.0),
        ]
    )
    img_inp_orig = to_float(image).unsqueeze(0)


    return img_inp_norm, img_inp_orig


class Normalize:
    """
    RGB3チャンネルについて指定された平均値・標準偏差で正規化するクラス
    TODO: クラスである必要があまりなさそうなので関数にしたい
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img: Image):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)
        proc_img[..., 0] = (imgarr[..., 0] / 255.0 - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255.0 - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255.0 - self.mean[2]) / self.std[2]
        return proc_img


def resize_image(img: Image, size: float, method: int) -> Image:
    shorter = min(img.size[1], img.size[0])
    ratio = size / shorter
    xsize = int(((ratio * img.size[0]) // 128) * 128)
    ysize = int(((ratio * img.size[1]) // 128) * 128)
    return img.resize((xsize, ysize), method)


normalizer = Normalize()


def load_image(image: Image) -> Tuple[torch.Tensor, torch.Tensor]:
    img_norm = normalizer(image)

    # change channel order HCW -> CHW
    img_orig = (np.transpose(image, (2, 0, 1)) / 255.0 - 0.5) * 2
    img_norm = np.transpose(img_norm, (2, 0, 1))

    # to torch.Tensor and add one batch channel
    img_norm = torch.tensor(img_norm).unsqueeze(0)
    img_orig = torch.tensor(img_orig, dtype=torch.float32).unsqueeze(0)
    return img_norm, img_orig

def save_image(image: Image, name: str):
    image.save(name)

@lru_cache()
def load_char_label_dicts(seperator="\x1f"):
    char_dict = dict()
    label_dict = dict()
    with open(get_char_dict_file_path(), "rt") as fr:
        for line in fr:
            sp = line.strip("\n").split(seperator)
            char_dict[int(sp[1])] = sp[0]
            label_dict[sp[0]] = int(sp[1])
    return char_dict, label_dict

@lru_cache()
def load_font_dicts():
    with open(get_google_font_list_filename(), "r") as f:
        font_list = f.read().splitlines()
    return {i: f"{get_google_font_path()}/{f}" for i, f in enumerate(font_list)}
