from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import json

from .AutoDriveDataset import AutoDriveDataset
from .classify import bdd_labels


class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, transform=None):
        super().__init__(cfg, is_train, transform)
        self.db = self._get_db()

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        """
        gt_db = []
        for index in range(len(self.label_list)):
            rec = []
            mask_path = self.mask_list[index]
            label_path = self.label_list[index]
            image_path = mask_path.replace(self.mask_root,self.img_root).replace(".png",".jpg")
            
            label = json.load(open(label_path))
            data = label['frames'][0]['objects']
            data = self.filter_data(data)
            gt = np.zeros(len(data), 5)
            for idx, obj in enumerate(data):
                gt[idx][0] = bdd_labels[obj["category"]]
                bbox = obj["box2d"]
                gt[idx][1:] = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                
            rec.append({
                'image': image_path,
                'label': gt,
                'mask': mask_path
            })

            gt_db.extend(rec)

        return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if obj.has_key('box2d'):
                remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass