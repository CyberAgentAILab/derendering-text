import traceback
from logzero import logger as log
from typing import List
from .sampler import Sampler
from .renderer import Renderer
from .data_handler import DataHandler
from src.dto.dto_generator import TextGeneratorInputHandler, TrainingFormatData
from .synthtext_lib import synthtext_util as stu


class GeneratorHandler(object):

    def __init__(
            self,
            load_data_path: str,
            max_time: int = None,
            lang: str = 'japanese'):
        self.sampler = Sampler(load_data_path, lang)
        self.renderer = Renderer()
        self.max_time = max_time

    def sample_and_render(
            self,
            ih: TextGeneratorInputHandler,
            dh: DataHandler):
        try:
            if self.max_time is None:
                placed = self.sampler.sample(ih, dh)
                if placed:
                    self.renderer.render(ih, dh)
            else:
                with stu.time_limit(self.max_time):
                    placed = self.sampler.sample(ih, dh)
                    if placed:
                        self.renderer.render(ih, dh)
            return placed
        except stu.TimeoutException:
            pass
            return False
        except BaseException:
            traceback.print_exc()
            # some error in placing text on the region
            return False

    def run_per_instance(self, ih: TextGeneratorInputHandler, dh: DataHandler):
        # get loop items
        # loop_num: the number of text for one image
        # reg_idx: region index
        # aug_idx: augmented id linked to region index
        reg_idx, aug_idx = ih.get_loop_items()
        loop_num = len(aug_idx)
        # initialize flag for text placement at least one text
        placed_oom = False
        # initialize data handler
        dh.sample_initialize()
        for idx in range(loop_num):
            # set mask for text placement
            ih.set_collision_mask(reg_idx, aug_idx, idx)
            # sample and render one text from data in ih
            # store rendering parameters to dh
            placed = self.sample_and_render(ih, dh)
            if placed:
                # update a collision mask in input handler if sampling is
                # success
                ih.update_collision_mask(
                    dh.box_alpha_mask, reg_idx, aug_idx, idx)
                placed_oom = True
        return placed_oom

    def run(self, ih: TextGeneratorInputHandler, ninstance: int = 1) -> List[TrainingFormatData]:
        res = []
        # generate ninstance number of text images for one background image
        for n in range(ninstance):
            canvas_img = ih.bg
            # data handler for outputs and rendering
            dh = DataHandler(canvas_img)
            # run
            placed_oom = self.run_per_instance(ih, dh)
            log.debug(f'placement {placed_oom}')
            if placed_oom:
                # export data with data object class format if sampling sucesses at least one text
                # format -> see TrainingFormatData() in
                # src/dto/dto_generator.py
                tfd = dh.export_training_format_data()
                res.append(tfd)
        return res
