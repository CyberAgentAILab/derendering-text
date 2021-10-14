import os
import random
import itertools
import signal
from contextlib import contextmanager
import numpy as np
import cv2
from PIL import Image
import scipy.stats as sstat
# http://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        # raise TimeoutException, colorize(Color.RED, "   *** Timed out!",
        # highlight=True)
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        # raise TimeoutException, colorize(Color.RED, "   *** Timed out!",
        # highlight=True)
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def load_synth_text_data(depth_db, seg_db, bg_dir, imname):
    img = Image.open(os.path.join(bg_dir, imname)).convert('RGB')
    depth = depth_db[imname][:].T
    depth = depth[:, :, 1]
    # get segmentation:
    seg = seg_db['mask'][imname][:].astype('float32')
    area = seg_db['mask'][imname].attrs['area']
    label = seg_db['mask'][imname].attrs['label']
    # re-size uniformly:
    sz = depth.shape[:2][::-1]
    img = np.array(img.resize(sz, Image.ANTIALIAS))
    seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))
    return img, depth, seg, area, label


def sample_weighted(p_dict):
    ps = np.array(list(p_dict.keys()))
    return p_dict[np.random.choice(ps, p=ps)]


def yx2xy(bbsyx: np.ndarray) -> np.ndarray:
    bbsxy = np.zeros_like(bbsyx)
    bbsxy[0, :, :] = bbsyx[1, :, :]
    bbsxy[1, :, :] = bbsyx[0, :, :]
    return bbsxy


def xy2yx(bbsxy: np.ndarray) -> np.ndarray:
    bbsyx = np.zeros_like(bbsxy)
    bbsyx[0, :, :] = bbsxy[1, :, :]
    bbsyx[1, :, :] = bbsxy[0, :, :]
    return bbsyx


def homographyBB(bbs, H, offset=None):
    """
    Apply homography transform to bounding-boxes.
    BBS: 2 x 4 x n matrix  (2 coordinates, 4 points, n bbs).
    Returns the transformed 2x4xn bb-array.

    offset : a 2-tuple (dx,dy), added to points before transfomation.
    """
    eps = 1e-16
    # check the shape of the BB array:
    t, f, n = bbs.shape
    assert (t == 2) and (f == 4)

    # append 1 for homogenous coordinates:
    bbs_h = np.reshape(np.r_[bbs, np.ones((1, 4, n))], (3, 4 * n), order='F')
    if offset is not None:
        bbs_h[:2, :] += np.array(offset)[:, None]

    # perpective:
    bbs_h = H.dot(bbs_h)
    bbs_h /= (bbs_h[2, :] + eps)

    bbs_h = np.reshape(bbs_h, (3, 4, n), order='F')
    return bbs_h[:2, :, :]


def warpHomography(src_mat, H, dst_size):
    dst_mat = cv2.warpPerspective(
        src_mat,
        H,
        dst_size,
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return dst_mat


def charBB2wordBB(charBB, text):
    """
    Converts character bounding-boxes to word-level
    bounding-boxes.

    charBB : 2x4xn matrix of BB coordinates
    text   : the text string

    output : 2x4xm matrix of BB coordinates,
                where, m == number of words.
    """
    #wrds = text.split()
    wrds = text
    bb_idx = np.r_[0, np.cumsum([len(w) for w in wrds])]
    wordBB = np.zeros((2, 4, len(wrds)), 'float32')

    for i in range(len(wrds)):
        cc = charBB[:, :, bb_idx[i]:bb_idx[i + 1]]
        # fit a rotated-rectangle:
        # change shape from 2x4xn_i -> (4*n_i)x2
        cc = np.squeeze(np.concatenate(
            np.dsplit(cc, cc.shape[-1]), axis=1)).T.astype('float32')
        rect = cv2.minAreaRect(cc.copy())
        box = np.array(cv2.boxPoints(rect))

        # find the permutation of box-coordinates which
        # are "aligned" appropriately with the character-bb.
        # (exhaustive search over all possible assignments):
        cc_tblr = np.c_[cc[0, :],
                        cc[-3, :],
                        cc[-2, :],
                        cc[3, :]].T
        perm4 = np.array(list(itertools.permutations(np.arange(4))))
        dists = []
        for pidx in range(perm4.shape[0]):
            d = np.sum(np.linalg.norm(box[perm4[pidx], :] - cc_tblr, axis=1))
            dists.append(d)
        wordBB[:, :, i] = box[perm4[np.argmin(dists)], :].T
    return wordBB

#################################
# text sampler
#################################


class TextSource(object):
    """
    Provides text for words, paragraphs, sentences.
    """

    def __init__(self, min_nchar, fn):
        """
        TXT_FN : path to file containing text data.
        """
        self.min_nchar = min_nchar
        self.fdict = {'WORD': self.sample_word,
                      'LINE': self.sample_line,
                      'PARA': self.sample_para}

        with open(fn, 'r') as f:
            self.txt = [l.strip() for l in f.readlines()]

        # distribution over line/words for LINE/PARA:
        self.p_line_nline = np.array([0.85, 0.10, 0.05])
        self.p_line_nword = [4, 3, 12]  # normal: (mu, std)
        self.p_para_nline = [1.0, 1.0]  # [1.7,3.0] # beta: (a, b), max_nline
        self.p_para_nword = [1.7, 3.0, 10]  # beta: (a,b), max_nword

        # probability to center-align a paragraph:
        self.center_para = 0.5

    def check_symb_frac(self, txt, f=0.35):
        """
        T/F return : T iff fraction of symbol/special-charcters in
                     txt is less than or equal to f (default=0.25).
        """
        return np.sum([not ch.isalnum() for ch in txt]) / (len(txt) + 0.0) <= f

    def is_good(self, txt, f=0.35):
        """
        T/F return : T iff the lines in txt (a list of txt lines)
                     are "valid".
                     A given line l is valid iff:
                         1. It is not empty.
                         2. symbol_fraction > f
                         3. Has at-least self.min_nchar characters
                         4. Not all characters are i,x,0,O,-
        """
        def is_txt(l):
            char_ex = ['i', 'I', 'o', 'O', '0', '-']
            chs = [ch in char_ex for ch in l]
            return not np.all(chs)
        return [(len(l) > self.min_nchar
                 and self.check_symb_frac(l, f)
                 and is_txt(l)) for l in txt]

    def center_align(self, lines):
        """
        PADS lines with space to center align them
        lines : list of text-lines.
        """
        ls = [len(l) for l in lines]
        max_l = max(ls)
        for i in range(len(lines)):
            l = lines[i].strip()
            dl = max_l - ls[i]
            lspace = dl // 2
            rspace = dl - lspace
            lines[i] = ' ' * lspace + l + ' ' * rspace
        return lines

    def get_lines(self, nline, nword, nchar_max, f=0.35, niter=100):
        def h_lines(niter=100):
            lines = ['']
            iter = 0
            while not np.all(self.is_good(lines, f)) and iter < niter:
                iter += 1
                line_start = np.random.choice(len(self.txt) - nline)
                lines = [self.txt[line_start + i] for i in range(nline)]
            return lines

        lines = ['']
        iter = 0
        while not np.all(self.is_good(lines, f)) and iter < niter:
            iter += 1
            lines = h_lines(niter=100)
            # get words per line:
            nline = len(lines)
            for i in range(nline):
                words = lines[i].split()
                dw = len(words) - nword[i]
                if dw > 0:
                    first_word_index = random.choice(range(dw + 1))
                    lines[i] = ' '.join(
                        words[first_word_index:first_word_index + nword[i]])

                # chop-off characters from end:
                while len(lines[i]) > nchar_max:
                    if not np.any([ch.isspace() for ch in lines[i]]):
                        lines[i] = ''
                    else:
                        lines[i] = lines[i][:len(
                            lines[i]) - lines[i][::-1].find(' ')].strip()

        if not np.all(self.is_good(lines, f)):
            return  # None
        else:
            return lines

    def sample(self, nline_max, nchar_max, kind='PARA'):
        return self.fdict[kind](nline_max, nchar_max)

    def sample_word(self, nline_max, nchar_max, niter=100):
        rand_line = self.txt[np.random.choice(len(self.txt))]
        words = rand_line.split()
        rand_word = random.choice(words)

        iter = 0
        while iter < niter and (not self.is_good([rand_word])[
                                0] or len(rand_word) > nchar_max):
            rand_line = self.txt[np.random.choice(len(self.txt))]
            words = rand_line.split()
            rand_word = random.choice(words)
            iter += 1

        if not self.is_good([rand_word])[0] or len(rand_word) > nchar_max:
            return []
        else:
            return rand_word

    def sample_line(self, nline_max, nchar_max):
        nline = nline_max + 1
        while nline > nline_max:
            nline = np.random.choice([1, 2, 3], p=self.p_line_nline)

        # get number of words:
        nword = [
            self.p_line_nword[2] *
            sstat.beta.rvs(
                a=self.p_line_nword[0],
                b=self.p_line_nword[1]) for _ in range(nline)]
        nword = [max(1, int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            return '\n'.join(lines)
        else:
            return []

    def sample_para(self, nline_max, nchar_max):
        # get number of lines in the paragraph:
        nline = nline_max * \
            sstat.beta.rvs(a=self.p_para_nline[0], b=self.p_para_nline[1])
        nline = max(1, int(np.ceil(nline)))

        # get number of words:
        nword = [
            self.p_para_nword[2] *
            sstat.beta.rvs(
                a=self.p_para_nword[0],
                b=self.p_para_nword[1]) for _ in range(nline)]
        nword = [max(1, int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            # center align the paragraph-text:
            if np.random.rand() < self.center_para:
                lines = self.center_align(lines)
            return '\n'.join(lines)
        else:
            return []
