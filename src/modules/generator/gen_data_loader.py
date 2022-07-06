import os
import random
from typing import Tuple, List
import cv2
import numpy as np
import scipy.signal as ssig
from PIL import Image
from logzero import logger as log


class SingleColorBG_Loader(object):
    def __init__(self):
        pass

    def load_bg_and_masks(self, index: int) -> Tuple[np.array, List[np.array]]:
        bg = np.zeros((512, 512, 3))
        if random.random() < 0.2:
            val = random.randint(230, 255)
            bg[:, :, 0] = val
            bg[:, :, 1] = val
            bg[:, :, 2] = val
        elif random.random() < 0.4:
            val = random.randint(0, 50)
            bg[:, :, 0] = val
            bg[:, :, 1] = val
            bg[:, :, 2] = val
        else:
            bg[:, :, 0] = random.randint(0, 255)
            bg[:, :, 1] = random.randint(0, 255)
            bg[:, :, 2] = random.randint(0, 255)
        h, w, c = bg.shape
        mask_random = np.zeros((h, w), dtype=np.uint8)
        return bg, [mask_random]

    def __len__(self):
        return 100000


class FMD_Loader(object):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.category_list = [
            "fabric",
            "foliage",
            "glass",
            "leather",
            "metal",
            "paper",
            "plastic",
            "stone",
            "water",
            "wood"]

    def get_box(self, mask: np.array, box_mask: np.array) -> Tuple[np.array,List,int,int]:
        out_arr = np.zeros_like(mask)
        mask[mask > 127] = 1e8
        intersect = ssig.fftconvolve(mask, box_mask[::-1, ::-1], mode='valid')
        safemask = intersect < 1e8
        minloc = np.transpose(np.nonzero(safemask))
        loc = minloc[np.random.choice(minloc.shape[0]), :]
        w, h = box_mask.shape
        out_arr[loc[0]:loc[0] + w, loc[1]:loc[1] + h] += box_mask
        return out_arr, loc, w, h

    def load_bg_and_masks(self, index: int) -> Tuple[np.array, List[np.array]]:
        category = self.category_list[random.randint(0, 9)]
        imid = random.randint(1, 50)
        if category == 'foliage':
            fn = os.path.join(
                self.data_dir, "image/{}/{}_final_{:0>3}_new.jpg".format(category, category, imid))
        else:
            fn = os.path.join(
                self.data_dir, "image/{}/{}_moderate_{:0>3}_new.jpg".format(category, category, imid))
        bg_org = np.array(Image.open(fn))
        if category == 'foliage':
            fn = os.path.join(
                self.data_dir, "mask/{}/{}_final_{:0>3}_new.jpg".format(category, category, imid))
        else:
            fn = os.path.join(
                self.data_dir, "mask/{}/{}_moderate_{:0>3}_new.jpg".format(category, category, imid))
        fmd_mask = cv2.imread(fn)[:, :, ::-1]
        fmd_mask = np.mean(fmd_mask, 2)
        box_mask = np.ones((256, 256), dtype=np.float32) * 255
        fmd_mask_inv = np.zeros_like(fmd_mask).astype(np.float32)
        fmd_mask_inv = 255 - fmd_mask.astype(np.float32)
        out, loc, w, h = self.get_box(fmd_mask_inv.astype(
            np.float32), box_mask.astype(np.float32))
        bg = bg_org[loc[0]:loc[0] + w, loc[1]:loc[1] + h, :]
        bg = cv2.resize(bg, (512, 512))
        mask_random = np.zeros(bg.shape[0:2], dtype=np.float32)
        mask_random = mask_random.astype(np.uint8)
        return bg, [mask_random]

    def __len__(self):
        return 100000


class Default_Loader(object):
    def __init__(
            self,
            bg_dir,
            mask_dir=None,
            bg_list=None,
            mask_list=None,
            bg_suffix='.jpg',
            mask_suffix='.png'):
        self.bg_dir = bg_dir
        self.mask_dir = mask_dir
        self.bg_suffix = bg_suffix
        self.mask_suffix = mask_suffix
        if bg_list is not None:
            f = open(bg_list, 'r')
            self.bg_list = f.read().splitlines()
            f.close()
        else:
            self.bg_list = os.listdir(self.bg_dir)
        if self.mask_dir is not None:
            if mask_list is not None:
                f = open(mask_list, 'r')
                self.mask_list = f.read().splitlines()
                f.close()
            else:
                self.mask_list = []
                for imname in self.bg_list:
                    prefix = imname.split('.')[0]
                    mask_name = f'{prefx}.{mask_suffix}'
                    self.mask_list.append(mask_name)

    def load_npz(self, file_name: str, th: float=0.5) -> np.array:
        mask = np.load(file_name)['arr_0'].astype(np.float32)
        loc = mask > 0.5
        mask[loc == 1] = 1
        mask[loc == 0] = 0
        return mask

    def load_img(self, file_name: str) -> np.array:
        return cv2.imread(file_name, cv2.IMREAD_UNCHANGED).astype(np.float32)

    def postprocess_for_mask(self, mask: np.array, h: int, w: int) -> np.array:
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return mask * 255

    def load_bg_and_masks(self, index):
        imgname = os.path.join(self.bg_dir, self.bg_list[index])
        bg = Image.open(imgname)
        bg = np.array(bg)[:, :, :3]
        h, w, c = bg.shape
        if self.mask_dir is not None:
            maskname = os.path.join(self.mask_dir, self.mask_list[index])
            if self.mask_suffix == 'png' or self.mask_suffix == 'jpg':
                mask = self.load_img(maskname)
            elif self.mask_suffix == 'npz':
                mask = self.load_npz(maskname)
            else:
                raise NotImplementedError(
                    'Not implemented mask data format. \
                    Change format to png, jpg, npz or configurate your own loader.'
                )
            mask = self.postprocess_for_mask(mask, h, w)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        return bg, [mask]

    def __len__(self):
        return len(self.bg_list)
