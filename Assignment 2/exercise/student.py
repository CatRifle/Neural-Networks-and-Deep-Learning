#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
import re
import string
# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    List, length = [], len(sample)
    
    for i in range(length):
        string = re.sub(r'[^a-zA-Z]', "", sample[i])

        if string:
            List.append(string)
    return List


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {'the','this','that','it','its','they','their','them','he'
                 ,'his','she','her','you','my','mine','i','have','service',
                 'manifestation','jobs','stores','shop','school','waiter',
                 'waitress','conversations'}
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """

    ratingOutput = torch.argmax(ratingOutput,dim=-1)
    categoryOutput = torch.argmax(categoryOutput,dim=-1)
    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(input_size=100, hidden_size=50, num_layers=2, batch_first=True, bidirectional=True,dropout=0.5)
        self.gru = tnn.GRU(input_size=300, hidden_size=50, num_layers=2, batch_first=True, bidirectional=True,dropout=0.9)
        self.linear1 = tnn.Linear(in_features=2*2*50, out_features=2)
        self.linear2 = tnn.Linear(in_features=2*2*50, out_features=5)


    def forward(self, input, length):
        #input.shape([batchSize,72,dim])
        #output,(h_n,c_n)=self.lstm(input)
        output,h_n=self.gru(input)
        #output of shape (seq_len, batch, num_directions * hidden_size)
        #h_n of shape (num_layers * num_directions, batch, hidden_size)
        #c_n of shape(num_layers * num_directions, batch, hidden_size)
        h_n = h_n.transpose(0,1)
        x = h_n.reshape(-1,2*2*50)
        #x = torch.cat((output[:, -1, :], output[:, 0, :]), dim=1)
        # print(x.shape)
        rating = self.linear1(x)
        #print(rating.shape)
        category = self.linear2(x)
        return rating,category

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss = tnn.CrossEntropyLoss()


    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingloss = self.loss(ratingOutput,ratingTarget)
        categoryloss = self.loss(categoryOutput,categoryTarget)
        ratio = 0.6
        resultloss = (1-ratio)*ratingloss + ratio*categoryloss
        return resultloss
        pass

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 5
optimiser = toptim.Adam(net.parameters(), lr=1e-3)
