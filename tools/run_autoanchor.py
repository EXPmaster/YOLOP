import os, sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from lib.utils import kmean_anchors
from lib.config import cfg
import torch
import torchvision.transforms as transforms
import lib.dataset as dataset


def run_anchor(dataset, thr=4.0, imgsz=640):
    na = 9
    new_anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)


if __name__ == '__main__':
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    run_anchor(dataset, imgsz=min(cfg.MODEL.IMAGE_SIZE))
