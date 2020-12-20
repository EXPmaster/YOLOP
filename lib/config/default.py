from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.NEED_AUTOANCHOR = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''
_C.MODEL.HEADS_NAME = ['']
_C.MODEL.PRETRAINED = ''
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)

# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.BOX_GAIN = 0.05  # box loss gain
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 1.0  # object loss gain


# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = ''
_C.DATASET.LABELROOT = ''
_C.DATASET.MASKROOT = ''
_C.DATASET.DATASET = ''
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_DATA = False

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add

# train
_C.TRAIN = CN(new_allowed=True)

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.VAL_FREQ = 10
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.MODEL_FILE = ''


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    # cfg.freeze()