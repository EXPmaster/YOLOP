# import os, sys
# import numpy as np
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
from lib.utils import kmean_anchors, check_anchor_order
from lib.config import cfg
import torchvision.transforms as transforms
import torch
import lib.dataset as dataset
from lib.utils import is_parallel


def run_anchor(dataset, model, thr=4.0, imgsz=640):
    det = model.module.model[model.module.detector_index] if is_parallel(model) \
        else model.model[model.detector_index]
    anchor_num = det.na * det.nl
    new_anchors = kmean_anchors(dataset, n=anchor_num, img_size=imgsz, thr=thr, gen=1000, verbose=False)
    new_anchors = torch.tensor(new_anchors, device=det.anchors.device).type_as(det.anchors)
    det.anchor_grid[:] = new_anchors.clone().view_as(det.anchor_grid)  # for inference
    det.anchors[:] = new_anchors.clone().view_as(det.anchors) / det.stride.to(det.anchors.device).view(-1, 1, 1)  # loss
    check_anchor_order(det)
    print('New anchors saved to model. Update model config to use these anchors in the future.')


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
    run_anchor(dataset, thr=cfg.TRAIN.ANCHOR_THRESHOLD, imgsz=min(cfg.MODEL.IMAGE_SIZE))
