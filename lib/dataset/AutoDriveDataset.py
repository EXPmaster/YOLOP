import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from visualization import plot_img_and_mask
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh


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
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        # self.label_list = self.label_root.iterdir()
        self.mask_list = self.mask_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)
    
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
        -image: transformed image, 先通过自己的数据增强(type:numpy), 然后使用self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        seg_label = cv2.imread(data["mask"], 0)
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
        
        (img, seg_label), ratio, pad = letterbox((img, seg_label), resized_shape, auto=False, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        ratio = (w / w0, h / h0)
        # print(resized_shape)
        
        det_label = data["label"]
        labels=[]
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w0 * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h0 * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w0 * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h0 * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
            
        if self.is_train:
            combination = (img, seg_label)
            (img, seg_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR
            )
            #print(labels.shape)
            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)

            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # if self.is_train:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                seg_label = np.fliplr(seg_label)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                seg_label = np.filpud(seg_label)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]
        
        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
        _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_label = torch.stack((seg2[0],seg1[0]),0)
        target = [labels_out, seg_label]
        img = self.transform(img)
        # if self.cfg.TRAIN.PLOT:
        #     if idx < 10:
        #         img_test = Image.open(data["image"])
        #         _, seg_mask = torch.max(seg_label, 0)
        #         seg_mask = seg_mask > 0.5
        #         # print(seg_mask.shape)
        #         plot_img_and_mask(img_test, seg_mask, idx)
        return img, target, data["image"], shapes

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

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det, label_seg = [], []
        for i, l in enumerate(label):
            l_det, l_seg = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
        return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(label_seg, 0)], paths, shapes

