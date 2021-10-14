import numpy as np
import pickle
import os
import sys
import traceback
from src.modules.generator import gen_mode
from src.dto.dto_generator import GeneratorDataInfo
from util.path_list import (
    get_generator_load_data_path,
    get_generator_save_data_path,
    get_fmd_data_dir,
)
from src.modules.generator.gen_data_loader import FMD_Loader, SingleColorBG_Loader, Default_Loader


def set_loader(args, gdi):
    if args.bgtype == 'synth_text':
        pass
    elif args.bgtype == 'fmd':
        gdi.set_loader(
            FMD_Loader(
                data_dir=get_fmd_data_dir(),
            )
        )
    elif args.bgtype == 'color':
        gdi.set_loader(
            SingleColorBG_Loader()
        )
    elif args.bgtype == 'bam':
        gdi.set_loader(
            Default_Loader(
                bg_dir=args.bg_dir,
                mask_dir=args.mask_dir,
                bg_list=args.bg_list,
                mask_list=args.mask_list,
                bg_suffix=args.bg_suffix,
                mask_suffix=args.mask_suffix,
            )
        )
    elif args.bgtype == 'book':
        gdi.set_loader(
            Default_Loader(
                bg_dir=args.bg_dir,
                mask_dir=args.mask_dir,
                bg_list=args.bg_list,
                mask_list=args.mask_list,
                bg_suffix=args.bg_suffix,
                mask_suffix=args.mask_suffix,
            )
        )
    elif args.bgtype == 'load':
        gdi.set_loader(
            Default_Loader(
                bg_dir=args.bg_dir,
                mask_dir=args.mask_dir,
                bg_list=args.bg_list,
                mask_list=args.mask_list,
                bg_suffix=args.bg_suffix,
                mask_suffix=args.mask_suffix,
            )
        )
    else:
        raise NotImplementedError(
            'Not implemented function. \
            Change options or configurate own setting.'
        )


def prepare_save_dirs(save_path, bg_dir, img_dir, alpha_dir, metadata_dir):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(bg_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(alpha_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)


def main(args):
    # load data information
    load_path = get_generator_load_data_path()
    save_path = os.path.join(
        get_generator_save_data_path(),
        f'{args.bgtype}_{args.lang}_{args.version}')
    bg_dir = os.path.join(save_path, 'bg')
    img_dir = os.path.join(save_path, 'img')
    alpha_dir = os.path.join(save_path, 'alpha')
    metadata_dir = os.path.join(save_path, 'metadata')
    prepare_save_dirs(save_path, bg_dir, img_dir, alpha_dir, metadata_dir)

    # set generator data information
    gdi = GeneratorDataInfo(
        load_path=load_path,
        save_path=save_path,
        bg_dir=bg_dir,
        img_dir=img_dir,
        alpha_dir=alpha_dir,
        metadata_dir=metadata_dir,
        prefixes=[],
    )
    # set loader to gdi
    set_loader(args, gdi)

    # generate text images
    if args.bgtype == 'synth_text':
        gen_mode.gen_with_synth_text_data(args, gdi)
    else:
        gen_mode.gen_with_simple_mask(args, gdi)

    # save data information
    saven = os.path.join(gdi.save_path, 'save_data_info.pkl')
    with open(saven, 'wb') as f:
        pickle.dump(gdi, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Skia-based text image generator')
    parser.add_argument(
        '--bgtype',
        type=str,
        default='synth_text',
        help='Specifying background type, which is also used for save name')
    parser.add_argument(
        '--version',
        type=str,
        default='tmp',
        help='for specifying save name')
    parser.add_argument('--lang', type=str, default='eng',
                        help='Currently supporting only english')
    parser.add_argument(
        '--secs_per_img',
        type=int,
        default=5,
        help='Maximum time for generating one text image')
    parser.add_argument(
        '--instance_per_img',
        type=int,
        default=1,
        help='The number of sampling for one background image')
    parser.add_argument(
        '--use_homography',
        type=bool,
        default=False,
        help='Option for use homography or not in synth text data format')
    parser.add_argument(
        '--bg_dir',
        type=str,
        default='src/modules/generator/example/bg',
        help='Directory of the background images for option of bgtype=load.')
    parser.add_argument(
        '--mask_dir',
        type=str,
        default='src/modules/generator/example/mask',
        help='Directory of the masks for option of bgtype=load.')
    parser.add_argument(
        '--bg_list',
        type=str,
        default='src/modules/generator/example/bg_list.txt',
        help='Name list of background images for option of bgtype=load.')
    parser.add_argument(
        '--mask_list',
        type=str,
        default='src/modules/generator/example/mask_list.txt',
        help='Name list of masks for option of bgtype=load.')
    parser.add_argument(
        '--bg_suffix',
        type=str,
        default='jpg',
        help='Suffix of background images for option of bgtype=load.')
    parser.add_argument(
        '--mask_suffix',
        type=str,
        default='png',
        help='Suffix of masks for option of bgtype=load.')
    args, _ = parser.parse_known_args()
    main(args)
