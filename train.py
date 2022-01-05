
import argparse
from logzero import logger as log
from src.modules.trainer import text_parser_trainer, inpaintor_trainer
from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Any, List, Tuple


class TrainTarget(IntEnum):
    TEXT_PARSER = 1
    INPAINTOR = 2


@dataclass
class DatasetConfig:
    data_dir: str
    # background images sometimes include raster texts (i.e., dataset includes
    # non-annotated texts)
    prior: float
    # background images sometimes include raster texts (i.e., dataset includes
    # non-annotated texts)
    noisy_bg_option: bool

# temporary setting
dataset_list = [
    DatasetConfig(
        data_dir='gen_data/color_eng_tmp',
        prior=1.0,
        noisy_bg_option=False,
    )
]

# text parser trainer setting in the paper
# dataset_list=[
#     DatasetConfig(
#         data_dir='DIR_SYNTHTEXTBG_DATASET',
#         prior=1.0,
#         noisy_bg_option=False,
#     )
#     DatasetConfig(
#         data_dir='DIR_FMDBG_DATASET',
#         prior=1.0,
#         noisy_bg_option=False,
#     )
#     DatasetConfig(
#         data_dir='DIR_COLOR_DATASET',
#         prior=1.0,
#         noisy_bg_option=False,
#     )
#     DatasetConfig(
#         data_dir='DIR_BOOKCOVERBG_DATASET',
#         prior=1.0,
#         noisy_bg_option=True,
#     )
#     DatasetConfig(
#         data_dir='DIR_BAMBG_DATASET',
#         prior=1.0,
#         noisy_bg_option=True,
#     )
# ]

# inpaintor trainer setting in the paper
# dataset_list=[
#     DatasetConfig(
#         data_dir='DIR_SYNTHTEXTBG_DATASET',
#         prior=1.0,
#         noisy_bg_option=False,
#     )
#     DatasetConfig(
#         data_dir='DIR_FMDBG_DATASET',
#         prior=1.0,
#         noisy_bg_option=False,
#     )
#     DatasetConfig(
#         data_dir='DIR_COLOR_DATASET',
#         prior=1.0,
#         noisy_bg_option=False,
#     )
# ]

def main(args):
    if args.mode == TrainTarget.TEXT_PARSER:
        log.info('train text parser')
        text_parser_trainer.train(args, dataset_list)
    elif args.mode == TrainTarget.INPAINTOR:
        inpaintor_trainer.train(args, dataset_list)
    else:
        NotImplementedError()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description='mode selector for trainer')
        parser.add_argument(
            '--mode',
            type=int,
            default=1,
            help='mode:1 training text parser | mode:2 training inpaintor')
        parser.add_argument(
            '--save_data_path',
            type=str,
            default='logs/',
            help='save data path for model and materials generated during trainings')
        parser.add_argument(
            '--model_id',
            type=str,
            default='tmp',
            help='model id for save')
        parser.add_argument(
            '--pret',
            type=str,
            default="weights/pret.pth",
            help='pretrined model for the text parser model.')
        parser.add_argument(
            '--batch_size',
            type=int,
            default=8,
            help='number of batch size.')
        parser.add_argument(
            '--text_pool_num',
            type=int,
            default=10,
            help='number of texts in an image to handle')
        parser.add_argument(
            '--gpuid',
            required=False,
            default=-1,
            help='-1: use all gpu, others: index of gpu for training',
            type=int)
        parser.add_argument(
            '--nworker',
            required=False,
            default=2,
            help='num workers for data loader',
            type=int)
        args = parser.parse_args()
        main(args)
    except Exception as e:
        log.exception("Unexpected error has occurred")
