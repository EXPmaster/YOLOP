import numpy as np
import torch
from sklearn import datasets
class SegmentationMetric(object):
    '''
    imgLabel [batch_size, height(144), width(256)]
    confusionMatrix [[0(TN),1(FP)],
                     [2(FN),3(TP)]]
    '''
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
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-12)
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
        IoU[np.isnan(IoU)] = 0
        mIoU = np.nanmean(IoU)
        return mIoU
    
    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        # print(imgLabel.shape)
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

    def matrix(self):
        return self.confusionMatrix


    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def test():
    seg_out = np.array([[[[1,0,1],[0,1,0],[1,0,1]],[[0,1,0],[1,0,1],[0,1,0]]]])
    seg_out1 = torch.from_numpy(seg_out)

    #print(seg_out1.shape)
    _,da_predict=torch.max(seg_out1, 1)
    print(da_predict)
    target = np.array([[[[0,0,1],[0,1,0],[1,0,1]],[[1,1,0],[1,0,1],[0,1,0]]]])
    target1 = torch.from_numpy(target)
    _,da_gt=torch.max(target1, 1)
    print(da_gt)

    da_predict = da_predict[:, :2, :]
    da_gt = da_gt[:, :2, :]

    print(da_predict.shape)
    print(da_gt.shape)

    da_metric = SegmentationMetric(2)
    da_metric.reset()
    da_metric.addBatch(da_predict.cpu(), da_gt.cpu())

    matrix = da_metric.matrix()
    da_acc = da_metric.pixelAccuracy()
    da_IoU = da_metric.IntersectionOverUnion()
    da_mIoU = da_metric.meanIntersectionOverUnion()

    print(matrix)
    print(da_acc)
    print(da_IoU)
    print(da_mIoU)

    intersection = np.diag(matrix)
    union = np.sum(matrix, axis=1) + np.sum(matrix, axis=0) - np.diag(matrix)
    print(intersection)
    print(union)
    da_metric.reset()
    da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
    print(da_metric.matrix())
    print(da_metric.pixelAccuracy())

    #print(da_predict.shape)
    """_,da_gt=torch.max(target, 1)
    da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
    da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
    da_metric = SegmentationMetric(2)
    da_metric.reset()
    da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
    da_acc = da_metric.pixelAccuracy()
    da_IoU = da_metric.IntersectionOverUnion()
    da_mIoU = da_metric.meanIntersectionOverUnion()"""


def test_dbscan():
    X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
                                      noise=.05)
    X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
               random_state=9)
    X = np.concatenate((X1, X2))
    print(X.shape)
    
if __name__ == '__main__':
    test_dbscan()