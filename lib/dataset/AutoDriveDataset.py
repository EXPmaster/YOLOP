from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset


class AutoDriveDataset(Dataset):
    """
    A general Dataset 用于实现一些通用服务
    """
    def __init__(self, cfg, is_train, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train: bool,判断是否是训练集
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.transform = transform
        self.img_root = cfg.DATASET.DATAROOT
        self.label_root = cfg.DATASET.LABELROOT
        self.mask_root = cfg.DATASET.MASKROOT
        self.image_set = cfg.DATASET.TRAIN_SET
        self.label_list = os.listdir(cfg.DATASET.LABELROOT)
        self.mask_list = os.listdir(cfg.DATASET.MASKROOT)

        self.db = []

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)

    
    def _get_db(self):
        """
        在子数据集上完成，不删
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        在子数据集上完成，不删
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        获取database里的图片和标签，并对他们做数据增强

        Inputs:
        -idx: 图片在self.db(database)(list)里的索引
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -input: transformed image, 先通过自己的数据增强(type:numpy), 然后使用self.transform
        -target: ground truth
        -meta: information about the image

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        seg_label = cv2.imread(data["mask"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = (img, seg_label)
        target = data["label"]
        input = ...
        target = ...
        meta = ...
        return input, target, meta

    def select_data(self, db):
        """
        可以用这个函数对数据集中的数据进行过滤，
        不需要使用的话可以直接把这个函数删掉

        Inputs:
        -db: (list)数据集

        Returns:
        -db_selected: (list)过滤后的数据集
        """
        db_selected = ...
        return db_selected

