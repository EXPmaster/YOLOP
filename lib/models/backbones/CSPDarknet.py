import torch
from torch import tensor
import torch.nn as nn
import sys
import math
# sys.path.append("lib/models")
# sys.path.append("lib/utils")

from ..common import SPP, Conv, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from torch.nn import Upsample
from lib.utils import check_anchor_order

from lib.utils import initialize_weights

CSPDarknet_s = [
[ -1, Focus, [3, 32, 3]],
[ -1, Conv, [32, 64, 3, 2]],
[ -1, BottleneckCSP, [64, 64, 1]],
[ -1, Conv, [64, 128, 3, 2]],
[ -1, BottleneckCSP, [128, 128, 3]],
[ -1, Conv, [128, 256, 3, 2]],
[ -1, BottleneckCSP, [256, 256, 3]],
[ -1, Conv, [256, 512, 3, 2]],
[ -1, SPP, [512, 512, [5, 9, 13]]],
[ -1, BottleneckCSP, [512, 512, 1, False]]
]

MCnet = [
[ -1, Focus, [3, 32, 3]],
[ -1, Conv, [32, 64, 3, 2]],
[ -1, BottleneckCSP, [64, 64, 1]],
[ -1, Conv, [64, 128, 3, 2]],
[ -1, BottleneckCSP, [128, 128, 3]],
[ -1, Conv, [128, 256, 3, 2]],
[ -1, BottleneckCSP, [256, 256, 3]],
[ -1, Conv, [256, 512, 3, 2]],
[ -1, SPP, [512, 512, [5, 9, 13]]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ -1, Conv,[512, 256, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1, 6], Concat, [1]],
[ -1, BottleneckCSP, [512, 256, 1, False]],
[ -1, Conv, [256, 128, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,4], Concat, [1]],
[ -1, BottleneckCSP, [256, 128, 1, False]],
[ -1, Conv, [128, 128, 3, 2]],
[ [-1, 14], Concat, [1]],
[ -1, BottleneckCSP, [256, 256, 1, False]],
[ -1, Conv, [256, 256, 3, 2]],
[ [-1, 10], Concat, [1]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ [17, 20, 23], Detect,  [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]],
[ 17, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [8, 2, 3, 1]] #segmentation output
]


class CSPDarknet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(CSPDarknet, self).__init__()
        layers, save= [], []
        self.nc = 13    #output category num
        self.detector_index = -1

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                detects, _ = self.forward(torch.zeros(1, 3, s, s))
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            """y = None if isinstance(x, list) else x.shape"""
            if isinstance(block, Detect):   # save detector result
                out.append(x)
            cache.append(x if block.index in self.save else None)
        m=nn.Sigmoid()
        out.append(m(x))
        # print(out[0][0].shape, out[0][1].shape, out[0][2].shape)
        return out
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_net(is_train, **kwargs):
    # bb_block_cfg = CSPDarknet_s
    m_block_cfg = MCnet
    model = CSPDarknet(m_block_cfg, **kwargs)
    # print('get_net')
    return model


if __name__ == "__main__":
    model = get_net(False)
    """for module in model.modules():
        print(module)"""
    input_ = torch.randn((1, 3, 1280, 736))
    pred = model(input_)
    print(pred[1].shape)    #segment
    for detect_res in pred[0]:
        print(detect_res.shape) #detect
    print(model.training)

