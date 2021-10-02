# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        renew_x = x.view(x.shape[0], -1)
        output = F.log_softmax(self.linear(renew_x), dim=1)
        return output

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.Linear1 = nn.Linear(28 * 28, 180)
        self.Linear2 = nn.Linear(180, 10)
        self.t = torch.nn.Tanh()

    def forward(self, x):
        renew_x = x.view(x.shape[0], -1)
        output1 = self.t(self.Linear1(renew_x))
        output2 = F.log_softmax(self.Linear2(output1), dim=1)
        return output2

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(1, 14, 5)
        self.conv2 = nn.Conv2d(14, 28, 5)
        self.Linear1 = nn.Linear(28 * 20 * 20, 600)
        self.Linear2 = nn.Linear(600, 10)
        self.r = torch.nn.ReLU()

    def forward(self, x):
        output1 = self.r(self.conv1(x))
        output1 = torch.max_pool2d(output1, 2, 2)

        output2 = self.r(self.conv2(output1))
        renew_output = output2.view(x.shape[0], -1)

        output3 = self.r(self.Linear1(renew_output))
        output4 = F.log_softmax(self.Linear2(output3), dim=1)


        return output4

    '''def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(1, 14, 5)
        self.conv2 = nn.Conv2d(14, 28, 5)
        self.conv3 = nn.Conv2d(28, 56, 5)
        self.conv4 = nn.Conv2d(56, 112, 5)
        self.Linear1 = nn.Linear(112 * 12 * 12, 600)
        self.Linear2 = nn.Linear(600, 10)
        self.r = torch.nn.ReLU()

    def forward(self, x):
        output1 = self.r(self.conv1(x))

        #output1 = torch.max_pool2d(output1, 2, 2)

        output2 = self.r(self.conv2(output1))
        output2 = self.r(self.conv3(output2))
        output2 = self.r(self.conv4(output2))

        renew_output = output2.view(x.shape[0], -1)

        output3 = self.r(self.Linear1(renew_output))

        output4 = F.log_softmax(self.Linear2(output3), dim=1)


        return output4
    '''