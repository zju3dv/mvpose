from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class LSRLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(LSRLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        num_class = inputs.size()[1]
        targets = self._class_to_one_hot(targets.data.cpu(), num_class)
        targets = Variable(targets.cuda())
        outputs = torch.nn.LogSoftmax()(inputs)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def _class_to_one_hot(self, targets, num_class):
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.FloatTensor(targets.size()[0], num_class)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets, 1 - self.epsilon)
        targets_onehot.add_(self.epsilon / num_class)
        return targets_onehot
