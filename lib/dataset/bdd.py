import os
import numpy as np
import json

from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict


class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, transform=None):
        super().__init__(cfg, is_train, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, x, y, w, h]
        """
        gt_db = []
        width, height = self.cfg.MODEL.IMAGE_SIZE
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
                category=obj['category']
                if (category == "traffic light"):
                    color = obj['attributes']['trafficLightColor']
                    category = "tl_" + color
                if category in id_dict.keys():
                    x1 = float(obj['box2d']['x1'])
                    y1 = float(obj['box2d']['y1'])
                    x2 = float(obj['box2d']['x2'])
                    y2 = float(obj['box2d']['y2'])
                    cls_id = id_dict[category]
                gt[idx][0] = cls_id
                box = convert((width,height),(x1,x2,y1,y2))
                gt[idx][1:] = list(box)
                
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