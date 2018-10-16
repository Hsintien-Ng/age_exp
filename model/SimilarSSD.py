import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarSSD(nn.Module):
    def __init__(self):
        super(SimilarSSD, self).__init__()
        self.base_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.base_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.base_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.base_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )
        self.base_conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU()
        )
        self.up_bilinear = nn.UpsamplingBilinear2d(size=(13, 13))
        self.pred1_conv = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.pred1_fc = nn.Sequential(
            nn.Linear(13 * 13 * 64, 256),
            nn.Linear(256, 100)
        )
        self.extras_conv1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.pred2_conv = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.pred2_fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 256),
            nn.Linear(256, 100)
        )
        self.extras_conv2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.pred3_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pred3_fc = nn.Sequential(
            nn.Linear(4 * 4 * 64, 256),
            nn.Linear(256, 100)
        )
        self.extras_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.pred4_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pred4_fc = nn.Sequential(
            nn.Linear(2 * 2* 64, 256),
            nn.Linear(256, 100)
        )
        self.extras_conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.pred5_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pred5_fc = nn.Sequential(
            nn.Linear(1 * 1 * 64, 256),
            nn.Linear(256, 100)
        )
        self.predM_conv = nn.Conv2d(64 * 5, 64, kernel_size=3, padding=1)
        self.predM_fc = nn.Sequential(
            nn.Linear(13 * 13 * 64, 256),
            nn.Linear(256, 100)
        )

    def forward(self, input):
        output = self.base_conv1(input)
        output = F.max_pool2d(output, (2, 2))

        output = self.base_conv2(output)
        output = F.max_pool2d(output, (2, 2))

        output = self.base_conv3(output)
        output = F.max_pool2d(output, (2, 2))

        output = self.base_conv4(output)
        output = F.max_pool2d(output, (2, 2), ceil_mode=True)

        pred_conv1 = self.base_conv5(output)
        pred_conv1_c = self.pred1_conv(pred_conv1)
        pred_conv1_flatten = pred_conv1_c.view(pred_conv1_c.size()[0], -1)
        pred1 = F.softmax(self.pred1_fc(pred_conv1_flatten), dim=-1)

        pred_conv2 = self.extras_conv1(pred_conv1)
        pred_conv2_c = self.pred2_conv(pred_conv2)
        pred_conv2_flatten = pred_conv2_c.view(pred_conv2_c.size()[0], -1)
        pred2 = F.softmax(self.pred2_fc(pred_conv2_flatten), dim=-1)
        upbili_conv2_c = F.interpolate(pred_conv2_c, size=(13, 13))

        pred_conv3 = self.extras_conv2(pred_conv2)
        pred_conv3_c = self.pred3_conv(pred_conv3)
        pred_conv3_flatten = pred_conv3_c.view(pred_conv3_c.size()[0], -1)
        pred3 = F.softmax(self.pred3_fc(pred_conv3_flatten), dim=-1)
        upbili_conv3_c = F.interpolate(pred_conv3_c, size=(13, 13))

        pred_conv4 = self.extras_conv3(pred_conv3)
        pred_conv4_c = self.pred4_conv(pred_conv4)
        pred_conv4_flatten = pred_conv4_c.view(pred_conv4_c.size()[0], -1)
        pred4 = F.softmax(self.pred4_fc(pred_conv4_flatten), dim=-1)
        upbili_conv4_c = F.interpolate(pred_conv4_c, size=(13, 13))

        pred_conv5 = self.extras_conv4(pred_conv4)
        pred_conv5_c = self.pred5_conv(pred_conv5)
        pred_conv5_flatten = pred_conv5_c.view(pred_conv5_c.size()[0], -1)
        pred5 = F.softmax(self.pred5_fc(pred_conv5_flatten), dim=-1)
        upbili_conv5_c = F.interpolate(pred_conv5_c, size=(13, 13))

        concat_layer = torch.cat([pred_conv1_c, upbili_conv2_c, upbili_conv3_c,
                                  upbili_conv4_c, upbili_conv5_c], 1)
        output = self.predM_conv(concat_layer)
        pred_convM_flatten = output.view(output.size()[0], -1)
        pred_main = F.softmax(self.predM_fc(pred_convM_flatten), dim=-1)

        return [pred1, pred2, pred3, pred4, pred5, pred_main]


def SimilarSSD_parameters_func(net, lr):
    return net.parameters()


if __name__ == '__main__':
    net = SimilarSSD()
    input = torch.zeros(10, 3, 200, 200)
    net.train()
    p1, p2, p3, p4, p5, pm = net(input)