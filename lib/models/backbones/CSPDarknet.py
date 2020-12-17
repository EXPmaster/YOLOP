import torch
import torch.nn as nn
import sys
sys.path.append("lib/models")
sys.path.append("lib/utils")

from common import SPP,Conv,Bottleneck,BottleneckCSP,Focus,Concat
from torch.nn import Upsample

from utils import initialize_weights

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
[ -1, BottleneckCSP, [256, 256, 1, False]]
[ -1, Conv, [128, 128, 3, 2]],
[ [-1, 14], Concat, [1]],
[ -1, BottleneckCSP, [256, 256, 1, False]]
[ -1, Conv, [256, 256, 3, 2]],
[ [-1, 10], Concat, [1]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ [17, 20, 23], Detect,]
# TODO
]


class CSPDarknet(nn.Module):
    def __init__(self, block_cfg,**kwargs):
        super(CSPDarknet, self).__init__()
        layers, save= [], []
        self.nc = 13
        for i, (f,m,args) in enumerate(block_cfg):
            m = eval(m) if isinstance(m, str) else m  # eval strings
            m_ = m(*args)
            m_.i, m_.f= i , f
            layers.append(m_)
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        self.model, self.save = nn.Sequential(*(layers)), sorted(save)
        self.names = [str(i) for i in range(self.nc)]
        initialize_weights(self)

    def forward(self,x):
        y = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]       #calculate concat$detect
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

def get_net(is_train, **kwargs):
    block_cfg = CSPDarknet_s
    model = CSPDarknet(block_cfg,**kwargs)
    return model

if __name__ == "__main__":
    model = get_net(False)
    for module in model.modules():
        print(module)
    input_ = torch.randn((1, 3, 1280, 720))
    pred = model(input_)
    print(pred.shape)
