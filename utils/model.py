import torch
import torch.nn as nn
from utils.figure import scatter_plot


def load_pretrained_func(path, net):
    assert isinstance(net, nn.Module)
    m = torch.load(path)
    net.load_state_dict(m)

    return net


def tbWriter(tb_writer, data_dict, global_step, task):
    """
    help to write some scalars or images to tensorboard writer.
    :param tb_writer: 
    :param data_dict: 
    :param global_step: 
    """
    prediction = data_dict['predict']
    label = data_dict['groudTruth']
    indicator = data_dict['indicator']
    loss = data_dict['loss']
    if task == 'Exp':
        indicatorName = 'train_accuracy'
    else:
        indicatorName = 'train_MAE'

    if isinstance(label, list):
        label = label[0]

    tb_writer.add_figure('predict',
                         scatter_plot(prediction.detach().cpu(), task), global_step)
    tb_writer.add_figure('groudTruth',
                         scatter_plot(label.cpu(), task), global_step)
    tb_writer.add_scalar(indicatorName, indicator.cpu(), global_step)
    tb_writer.add_scalar('loss', loss.item(), global_step)