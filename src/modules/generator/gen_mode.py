import os
import h5py
import pickle
import traceback
from typing import List
import numpy as np
import cv2
from logzero import logger as log
from src.dto.dto_generator import GeneratorDataInfo, TrainingFormatData, TextGeneratorInputHandler
from .src.handler import GeneratorHandler
from .src.synthtext_lib import synthtext_util as stu
from util.path_list import get_generator_load_data_path, get_generator_save_data_path


def add_res_to_db(gdi: GeneratorDataInfo, res: List[TrainingFormatData], prefix: int) -> None:
    for i in range(len(res)):
        prefix_instance = f"{prefix}_{i}"
        # save large volume elements with different format like image and alpha
        # maps
        bg_name = os.path.join(gdi.bg_dir, f'{prefix_instance}.jpg')
        cv2.imwrite(bg_name, res[i].bg[:, :, ::-1])
        img_name = os.path.join(gdi.img_dir, f'{prefix_instance}.jpg')
        cv2.imwrite(img_name, res[i].img[:, :, ::-1])
        alpha_name = os.path.join(gdi.alpha_dir, f'{prefix_instance}.npz')
        np.savez_compressed(alpha_name, res[i].alpha)
        # delete large volume elements from pickle format data
        res[i].del_large_volume_elements()
        res[i].set_deleted_file_names(bg_name, img_name, alpha_name)
        saven = os.path.join(gdi.metadata_dir, f'{prefix_instance}.pkl')
        with open(saven, 'wb') as f:
            pickle.dump(res[i], f)
        gdi.prefixes.append(prefix_instance)


def gen_with_synth_text_data(args, gdi: GeneratorDataInfo) -> None:
    # load synth text data
    #db = h5py.File(os.path.join(get_generator_load_data_path(),'dset_8000.h5'),'r')
    depth_db = h5py.File(
        os.path.join(
            get_generator_load_data_path(),
            'depth.h5'),
        'r')
    seg_db = h5py.File(
        os.path.join(
            get_generator_load_data_path(),
            'seg.h5'),
        'r')
    bg_dir = os.path.join(get_generator_load_data_path(), 'bg_img')
    img_names = sorted(depth_db.keys())
    start_idx,end_idx = 0, min(10000,len(img_names))
    # generator handler
    GH = GeneratorHandler(
        get_generator_load_data_path(),
        max_time=args.secs_per_img,
        lang=args.lang)
    if not args.use_homography:
        use_homography = False
    else:
        use_homography = True  # original synth text configuration

    for i in range(start_idx, end_idx):
        img_name = img_names[i]
        try:
            log.debug(f"{i}, {start_idx}, {end_idx}")
            # synth text data
            # bg: background image
            # depth, seg, area and label: for computing text placement area and
            # homography
            bg, depth, seg, area, label = stu.load_synth_text_data(
                depth_db, seg_db, bg_dir, img_name)
            # input data handler
            ih = TextGeneratorInputHandler(
                bg=bg,
                instance_per_img=args.instance_per_img,
                use_homography=use_homography
            )
            # set synth text data to input data handler
            ih.set_synth_text_inputs(
                depth,
                seg,
                area,
                label
            )
            # run generator
            res = GH.run(ih)
            if len(res) > 0:
                # save data
                prefix = img_name.split('.')[0]
                add_res_to_db(gdi, res, prefix)
        except BaseException:
            traceback.print_exc()
            continue
    depth_db.close()
    seg_db.close()


def gen_with_simple_mask(args, gdi: GeneratorDataInfo) -> None:
    # open databases:
    start_idx, end_idx = 0, min(10000, len(gdi.loader))
    GH = GeneratorHandler(
        get_generator_load_data_path(),
        max_time=args.secs_per_img,
        lang=args.lang)
    for i in range(start_idx, end_idx):
        try:
            log.debug(f"{i}, {start_idx}, {end_idx}")
            # input data
            # bg: background image
            # masks: List of mask for text placement
            bg, masks = gdi.load_bg_and_masks(i)
            # input data handler
            ih = TextGeneratorInputHandler(
                bg=bg,
                instance_per_img=args.instance_per_img,
                use_homography=False
            )
            ih.set_mask(masks)
            # run generator
            res = GH.run(ih)
            if len(res) > 0:
                # save data
                prefix = i
                add_res_to_db(gdi, res, prefix)
        except BaseException:
            traceback.print_exc()
            continue
