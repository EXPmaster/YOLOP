import torch.nn as nn


def get_net(cfg):
    """
    build up the whole network

    Inputs:
    -cfg: configurations

    Returns:
    -net

    More:
    use the base_net name and head_nets name in cfg 调用函数构建网络
    base_net: backbone
    head_nets: (list) all the heads may be used
    you can use if... else... or other structure to finish
    """
    base_net = ...
    head_nets = ...
    net = Shell(base_net, head_nets)
    return net


class Shell(nn.Module):
    def __init__(self, base_net, head_nets):
        super().__init__()

        self.base_net = base_net
        self.head_nets = nn.ModuleList(head_nets)

    def forward(self, x):
        x = self.base_net(x)
        head_outputs = [hn(x) for hn in self.head_nets]

        return head_outputs
