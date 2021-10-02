# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.linear1 = nn.Linear(2, num_hid)
        self.linear2 = nn.Linear(num_hid, 1)
        self.t = torch.nn.Tanh()
        self.s = torch.nn.Sigmoid()
        #self.r = torch.nn.ReLU()

    def forward(self, input):
        input[:, 0], input[:, 1] = torch.norm(input, p=2, dim=-1), torch.atan2(input[:, 1], input[:, 0])
        self.hid1 = self.t(self.linear1(input))

        output2 = self.s(self.linear2(self.hid1))
        return output2

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.linear1 = nn.Linear(2, num_hid)
        self.linear2 = nn.Linear(num_hid, num_hid)
        self.linear3 = nn.Linear(num_hid, 1)
        self.t = torch.nn.Tanh()
        #self.r = torch.nn.ReLU()
    def forward(self, input):
        self.hid1 = self.t(self.linear1(input))
        self.hid2 = self.t(self.linear2(self.hid1))
        output3 = torch.sigmoid(self.linear3(self.hid2))
        return output3

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        self.linear1_2 = nn.Linear(2, num_hid)
        self.linear1_3 = nn.Linear(2, num_hid)
        self.linear1_4 = nn.Linear(2, 1)
        self.linear2_3 = nn.Linear(num_hid, num_hid)
        self.linear2_4 = nn.Linear(num_hid, 1)
        self.linear3_4 = nn.Linear(num_hid, 1)
        self.t = torch.nn.Tanh()

    def forward(self, input):
        self.hid1 = self.t(self.linear1_2(input))

        self.hid2 = self.t(self.linear1_3(input) + self.linear2_3(self.hid1))

        output3 = torch.sigmoid(self.linear1_4(input) + self.linear2_4(self.hid1) + self.linear3_4(self.hid2))

        return output3



def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batch norm, dropout back again

        if layer == 1:
            output = net.hid1[:, node]
        elif layer == 2:
            output = net.hid2[:, node]

        pred = (output >= 0).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')

