import torch
import torch.nn as nn


def load_pretrained_func(path, net):
    assert isinstance(net, nn.Module)
    m = torch.load(path)
    net.load_state_dict(m)

    return net