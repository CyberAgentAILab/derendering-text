import io
import time
import os
import torch
from logzero import logger as log
import argparse
from PIL import Image
import numpy as np
import pickle

from src.io import transform_inputs
from src.io import save_image
from src.models.model import Model
from src.modules.postprocess.vector import vectorize, vectorize_postref
from src.modules.postprocess.renderer import render_vd
from util.path_list import get_weight


def load_model(dev: torch.device):
    model = Model(dev).to(dev)
    model.load_state_dict(torch.load(get_weight()), strict=True)
    model.eval()
    return model

def test(imgfile, savepath, saveprefix, use_cpu=False, gpuid=0, use_postref=True, iter_count=200):
    # show device selectbox
    if use_cpu==True:
        dev = torch.device(f"cpu")
    else:
        dev = torch.device(f"cuda:{gpuid}")
    model = load_model(dev)

    # load selected image
    img_norm, img_orig, pil_img = transform_inputs(imgfile, 640)
    img_norm, img_orig = img_norm.to(dev), img_orig.to(dev)
    img_size = torch.tensor([pil_img.size[1], pil_img.size[0]]).unsqueeze(0)
    start = time.time()
    with torch.no_grad():
        outs = model(img_norm, img_orig)
    end = time.time()
    elapsed_model = end - start

    len_results = len(outs[0].bbox_information.get_text_rectangle()[0])
    if len_results == 0:
        log.warning("No text found")
        return

    # make vectorize inputs and show results
    inps = (img_norm, None, img_size)

    # w/o postref
    start = time.time()
    vd, rec_img = vectorize(pil_img, inps, outs)
    end = time.time()
    elapsed_vectorize = end - start

    # w/ postref
    if use_postref:
        start = time.time()
        vd, rec_img = vectorize_postref(
            pil_img, inps, outs, model.reconstractor, iter_count, dev=dev
        )
        end = time.time()
    # rendering
    output_img = render_vd(vd)
    # save
    save_image(
        Image.fromarray(output_img),
        os.path.join(
            savepath,
            f'{saveprefix}.jpg'))
    rec_img = torch.max(
        torch.min(
            rec_img,
            torch.zeros_like(rec_img) +
            255),
        torch.zeros_like(rec_img))
    save_image(
        Image.fromarray(
            rec_img.data.cpu().numpy()[0].transpose(
                1, 2, 0).astype(
                np.uint8)), os.path.join(
                    savepath, f'{saveprefix}_rec.jpg'))
    with open(os.path.join(savepath, f'{saveprefix}.pkl'), mode='wb') as f:
        pickle.dump(vd, f)

def main(args):
    test(
        imgfile = args.imgfile,
        savepath = args.savepath,
        saveprefix = args.saveprefix,
        use_cpu = args.use_cpu,
        gpuid = args.gpuid
    )
    


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Test script")
        parser.add_argument('--imgfile', required=False,
                            default="example/sample.jpg",
                            help='input image path',
                            type=str)
        parser.add_argument('--savepath', required=False,
                            default="res",
                            help='path for results',
                            type=str)
        parser.add_argument('--saveprefix', required=False,
                            default="sample",
                            help='prefix for save data',
                            type=str)
        parser.add_argument('--use_cpu', required=False,
                            default=False,
                            help='use cpu or not',
                            type=bool)
        parser.add_argument('--gpuid', required=False,
                            default=0,
                            help='gpu id',
                            type=int)
        args = parser.parse_args()
        os.makedirs(args.savepath, exist_ok=True)
        main(args)
    except Exception as e:
        log.exception("Unexpected error has occurred")
