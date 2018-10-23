import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.convp1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.convp2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.convp3 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.convp4 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.convp5 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, input):
        face, p1, p2, p3, p4, p5 = input
        output = self.conv1(face) # 96 16
        output = F.max_pool2d(output, (2, 2)) # 48 16

        output = self.conv2(output) # 48 32
        p0 = F.max_pool2d(output, (2, 2)) # 24 32

        p1 = self.convp1(p1)  # 24 32
        p2 = self.convp2(p2)  # 24 32
        p3 = self.convp3(p3)  # 24 32
        p4 = self.convp4(p4)  # 24 32
        p5 = self.convp5(p5)  # 24 32

        fusion = torch.cat([p0, p1, p2, p3, p4, p5], 1) # 24 192
        output = self.conv3(fusion) # 24 512
        output = F.avg_pool2d(output, (24, 24)) # 1 512

        output = output.view(output.size()[0], -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)

        output = F.softmax(output, dim=1)

        return output


def fusion_parameters_func(net, lr):

    return net.parameters()