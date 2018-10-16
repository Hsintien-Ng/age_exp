import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def scatter_plot(tensor, ylabel_name):
    """
    return a 2-D scatter plot for the tensor
    :param tensor: shape-[n] where n is the batch_size
    :return: a figure of scatter plot
    """
    y = tensor.numpy()
    y = y.astype(np.int32)
    x = np.linspace(1, tensor.shape[0], tensor.shape[0])
    x = x.astype(np.int32)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Scatter Plot')
    plt.ylim(0, 100)
    plt.xlabel('Sample')
    plt.ylabel(ylabel_name)
    ax.scatter(x, y, c='r', marker='o')

    # annotate
    for i, value in enumerate(y):
        ax.annotate(str(value), (x[i], y[i]))

    return fig


if __name__ == '__main__':
    tensor = torch.Tensor([50, 24, 32, 18])
    fig = scatter_plot(tensor, 'test')