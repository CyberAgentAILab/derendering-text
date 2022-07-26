import os
import pickle
import random
import numpy as np
import skia
from fontTools.ttLib import TTFont
from ..synthtext_lib.synthtext_util import TextSource
from src.dto.dto_generator import TextGeneratorInputHandler
from ..data_handler import DataHandler
from src.skia_lib import skia_util as sku
from util.path_list import (
    get_google_font_path,
    get_google_font_list_filename,
    get_newsgroup_text_courpas,
)


class TextFontSampler(object):
    def __init__(self, load_data_path: str, lang: str = 'jp'):
        # distribution over the type of text:
        # whether to get a single word, paragraph or a line:
        self.max_paircheck_trials = 100
        self.p_text = {0.0: 'WORD',
                       0.0: 'LINE',
                       1.0: 'PARA'}
        self.lang = lang
        self.min_nchar = 2
        if lang == 'eng':
            self.font_path = get_google_font_path()
            font_file_name = get_google_font_list_filename()
            self.text_source = TextSource(min_nchar=self.min_nchar,
                                          fn=get_newsgroup_text_courpas())
        else:
            raise NotImplementedError()

        f = open(font_file_name, 'r')
        self.font_list = f.read().splitlines()
        f.close()

    def sample_from_ad_text(self, nchar: int):
        nchar = max(int(nchar * (0.5 + random.random())), 2)
        flag = 0
        while(flag == 0):
            if 'text_num_{}'.format(nchar) in self.ad_text_list.keys():
                text_list = self.ad_text_list['text_num_{}'.format(nchar)]
                flag = 1
            else:
                nchar = max(int(nchar * random.random()), 2)
        return text

    def check_valid_text_font_pair(self, text_raw: str, font_path: str):
        text_split_by_newline = text_raw.split('\n')
        text = ''.join(text_split_by_newline)
        ttfont = TTFont(font_path)
        flag_num = 0
        for c in text:
            flag = 0
            for table in ttfont['cmap'].tables:
                for char_code, glyph_name in table.cmap.items():
                    if char_code == ord(c):
                        flag = 1  # font file includes glyphs for the character
            flag_num += flag
        # check whether it's possible to render the all characters with the
        # paired font
        check_result = len(text) == flag_num
        return check_result

    def font_sample(self):
        l = len(self.font_list)
        font_id = random.randint(0, l - 1)
        font_path = os.path.join(self.font_path, self.font_list[font_id])
        return font_id, font_path

    def text_sample(self, nline: int, nchar: int):
        if self.lang == 'eng':
            def sample_weighted(p_dict):
                ps = np.array(list(p_dict.keys()))
                return p_dict[np.random.choice(ps, p=ps)]
            text_type = sample_weighted(self.p_text)
            text_raw = self.text_source.sample(nline, nchar, text_type)
        else:
            raise NotImplementedError()
        return text_raw

    def get_valid_text_and_font(self, nline: int, nchar: int):
        for i in range(self.max_paircheck_trials):
            text_raw = self.text_sample(nline, nchar)
            if len(text_raw) == 0 or np.any(
                    [len(line) == 0 for line in text_raw]):
                continue
            font_id, font_path = self.font_sample()
            if self.check_valid_text_font_pair(text_raw, font_path):
                break
        if text_raw == []:
            return None, (None, None)
        
        texts = text_raw.split(os.linesep)
        return texts, (font_id, font_path)

    def get_nline_nchar(
            self,
            H: int,
            W: int,
            font_height: int,
            font_width: int):
        nline = int(np.ceil(H / (1.5 * font_height)))
        nchar = int(np.floor(W / font_width))
        return nline, nchar

    def get_nline_nchar_handler(self, H: int, W: int, font_size: int):
        if self.lang == 'eng':
            nline, nchar = self.get_nline_nchar(
                H, W, font_size, int(font_size * 0.4))
        else:
            raise NotImplementedError()
        if random.random() < 0.5:
            nline = 1
        return nline, nchar

    def sample(self, ih: TextGeneratorInputHandler, dh: DataHandler):
        # load data from data handler
        font_size, H, W = dh.tmp.get_data_for_text_font_sampler()
        # get the number of text line and character number
        nline, nchar = self.get_nline_nchar_handler(H, W, font_size)
        # get the pair of text and font
        texts, (font_id, font_path) = self.get_valid_text_and_font(nline, nchar)
        font_object = sku.load_font_by_skia_format(font_size, font_path)
        if texts is None:
            return None
        else:
            # set sampled data to data handler
            dh.tmp.set_text_font_sampler_data(
                texts, font_path, nline, font_object)
            return texts, font_size, font_id, font_path
