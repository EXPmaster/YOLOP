import torch

height, weight = 512, 512
pad_h, pad_w = 112, 0
predict = torch.randn(1,height,weight)
print(predict.shape)
predict = predict[:,pad_h:height-pad_h,pad_w:weight-pad_w]
print(predict.shape)