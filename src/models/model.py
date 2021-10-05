import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.inpaintor import Inpaintor
from src.models.reconstructor import Reconstructor

from .vectorization import Vectorization


class Model(nn.Module):
    def __init__(self, dev: torch.device = None):
        super().__init__()
        v = Vectorization(dev=dev)
        self.backbone = v.backbone
        self.down = v.down
        self.text_parser = v.text_parser
        self.inpaintor = Inpaintor(dev=dev)
        self.reconstractor = Reconstructor(dev=dev)

    def forward(self, im, img_org):
        features = self.backbone(im)
        _, features = self.down(features)
        text_information = self.text_parser(features, img_org)
        inpaint = self.inpaintor(img_org, text_information)
        inpaint = F.interpolate(inpaint, img_org.shape[2:4], mode="bilinear")
        rec = self.reconstractor(features, img_org, inpaint, text_information)
        return text_information, inpaint, rec
