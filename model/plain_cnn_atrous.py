import torch.nn as nn
import torch.nn.functional as F


class PlainCNN_Atrous(nn.Module):
    def __init__(self):
        super(PlainCNN_Atrous, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=5, dilation=5),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=2, dilation=2),
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
        self.fc1 = nn.Linear(14 * 14 * 128, 1000)
        self.fc2 = nn.Linear(1000, 8)

    def forward(self, input):
        output = self.conv1(input)  # 224, 64
        output = F.max_pool2d(output, (2, 2))  # 112, 64

        output = self.conv2(output)  # 112, 128
        output = F.max_pool2d(output, (2, 2))  # 56, 128

        output = self.conv3(output)  # 56, 256
        output = F.max_pool2d(output, (2, 2))  # 28, 256

        output = self.conv4(output)  # 28, 512
        output = F.max_pool2d(output, (2, 2))  # 14, 512

        output = self.conv5(output)  # 14, 256

        output = self.conv6(output)  # 14, 128

        output = output.view(output.size()[0], -1)
        output = F.relu(self.fc1(output))  # 1000
        output = self.fc2(output)  # 8

        output = F.softmax(output)

        return output
