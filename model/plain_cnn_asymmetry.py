import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainCNN_Asym(nn.Module):
    def __init__(self):
        super(PlainCNN_Asym, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (11, 1), padding=(5, 0)),
            nn.Conv2d(64, 64, (1, 11), padding=(0, 5)),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (7, 1), padding=(3, 0)),
            nn.Conv2d(128, 128, (1, 7), padding=(0, 3)),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (5, 1), padding=(2, 0)),
            nn.Conv2d(256, 256, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.fc1 = nn.Linear(13 * 13 * 128, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 100)

    def forward(self, input):
        output = self.conv1(input)  # 200, 64
        output = F.max_pool2d(output, (2, 2))  # 100, 64

        output = self.conv2(output)  # 100, 128
        output = F.max_pool2d(output, (2, 2))  # 50, 128

        output = self.conv3(output)  # 50, 256
        output = F.max_pool2d(output, (2, 2))  # 25, 256

        output = self.conv4(output)  # 25, 512
        output = F.max_pool2d(output, (2, 2), ceil_mode=True)  # 13, 512

        output = self.conv5(output)  # 13, 256

        output = self.conv6(output)  # 13, 128

        output = output.view(output.size()[0], -1)
        output = F.relu(self.fc1(output))  # 1000
        output = F.relu(self.fc2(output))  # 8
        output = self.fc3(output)

        output = F.softmax(output, dim=1)

        return output


def plain_parameters_func(net, lr):
    return net.parameters()


if __name__ == '__main__':
    net = PlainCNN_Asym()
    input = torch.zeros(10, 3, 200, 200)
    net.train()
    output = net(input)
    print(output.shape)
    print(output)