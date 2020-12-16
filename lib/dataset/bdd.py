from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from .AutoDriveDataset import AutoDriveDataset


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
        gt_db = ...
        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass