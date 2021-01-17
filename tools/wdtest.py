

"""if __name__ == '__main__':

    import torch
    a=torch.Tensor([[1,1,2,2],[1,1,3.100001,3],[1,1,3.1,3]])
    b=torch.Tensor([0.9,0.98,0.980005])

    from torchvision.ops import nms

    ccc=nms(a,b,0.4)
    print(ccc)
    print(a[ccc])
    """

import numpy as np
import cv2
__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N

P      TP    FP

N      FN    TN

"""
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from lib.models import get_net
from lib.core.general import non_max_suppression
import torch

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == "__main__":
    model = get_net(False)
    #for module in model.modules():
    #    print(module)

    input_ = torch.randn((24, 3, 144, 256))
    gt_image = torch.randn((4, 2, 144, 256))
    model.eval()
    #print(model.training)
    pred = model(input_)
    #print(pred[1].shape)    #segment:[1, 2, 1280, 736]
    inf_out , train_out = pred[0]
    _,predict=torch.max(pred[1], 1)
    _,gt=torch.max(gt_image, 1)
    metric = SegmentationMetric(2)
    metric.addBatch(predict, gt)
    acc = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print(acc, mIoU,FWIoU)
    print(predict.shape)
    #print(inf_out.shape)    #inf:[1, 57960, 85]
    #print(type(train_out))  #train: list/[[1, 3, 160, 92, 85],[1, 3, 80, 46, 85],[1, 3, 40, 23, 85]]
    mean=0
    for (i,g) in zip(pred[1],gt_image):
        _,predict=torch.max(i, 0)
        _,gt=torch.max(g, 0)
        metric = SegmentationMetric(2)
        metric.addBatch(predict, gt)
        acc = metric.pixelAccuracy()
        classAcc = metric.classPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
        mean+=acc
        print(acc, classAcc,mIoU,FWIoU)
    print(mean/4)

    """for t in train_out:
        print(t.shape)"""
    

    """output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.6)
    print(output)"""

"""    
    pre_image=
    
    metric = SegmentationMetric(4)
    metric.addBatch(pre_image, gt_image)
    acc = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print(acc, mIoU,FWIoU)"""

