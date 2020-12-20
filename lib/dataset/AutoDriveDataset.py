from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import random
import torch
import os
from torch.utils.data import Dataset
from utils import letterbox, augment_hsv, random_perspective, xyxy2xywh


class AutoDriveDataset(Dataset):
    """
    A general Dataset 用于实现一些通用服务
    """
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
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
        self.cfg = cfg.DATASET
        self.transform = transform
        self.inputsize = inputsize
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
        resized_shape = self.inputsize
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        w, h = img.shape[:2]
        image, ratio, pad = letterbox(img, resized_shape, auto=False, scaleup=True)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        
        det_label = data["label"]
        if det_label.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = det_label.copy()
                labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]

        if self.is_train:
            combination = (image, seg_label)
            img, seg_label, labels = random_perspective(combination, labels,
                                            degrees=self.cfg.ROT_FACTOR,
                                            translate=self.cfg.TRANSLATE,
                                            scale=self.cfg.SCALE_FACTOR,
                                            shear=self.cfg.SHEAR)
            augment_hsv(img, hgain=self.cfg.HSV_H, sgain=self.cfg.HSV_S, vgain=self.cfg.HSV_V)
        
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            if self.is_train:
                # random left-right flip
                lr_flip = True
                if lr_flip and random.random() < 0.5:
                    img = np.fliplr(img)
                    if len(labels):
                        labels[:, 1] = 1 - labels[:, 1]

                # random up-down flip
                ud_flip = False
                if ud_flip and random.random() < 0.5:
                    img = np.flipud(img)
                    if len(labels):
                        labels[:, 2] = 1 - labels[:, 2]

            labels_out = torch.zeros((len(labels), 6))
            if len(labels):
                labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        target = [labels, seg_label]
        
        return img, target

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

