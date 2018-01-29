import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2h.register_backward_hook(self.depthDecay)
        self.depth = 0
        self.grad_weight = torch.zeros(hidden_size, input_size + hidden_size).cuda()
        self.grad_bias = torch.zeros(hidden_size).cuda()
        self.decay_alpha = 1.00
        
    def forward(self, input, hidden):
        self.depth = 0
        self.grad_weight = torch.zeros(self.hidden_size, self.input_size + self.hidden_size).cuda()
        self.grad_bias = torch.zeros(self.hidden_size).cuda()
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        return hidden

    def initHidden(self, bsz):
        return Variable(torch.zeros(bsz, self.hidden_size)).cuda()

    def depthDecay(self, module, grad_input, grad_output):
        self.depth += 1
        print("depth", self.depth)
        print(grad_input[2].data.t().mean())
        self.grad_weight += grad_input[2].data.t() * (self.decay_alpha**self.depth)
        self.grad_bias += grad_input[0].data.sum(0) * (self.decay_alpha**self.depth)
        return grad_input