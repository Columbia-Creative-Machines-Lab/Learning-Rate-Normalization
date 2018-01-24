import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.igate = nn.Linear(input_size + hidden_size, hidden_size)
        self.fgate = nn.Linear(input_size + hidden_size, hidden_size)
        self.ggate = nn.Linear(input_size + hidden_size, hidden_size)
        self.ogate = nn.Linear(input_size + hidden_size, hidden_size)
        self.igate.register_backward_hook(self.iDepthDecay)
        self.fgate.register_backward_hook(self.fDepthDecay)
        self.ggate.register_backward_hook(self.gDepthDecay)
        self.ogate.register_backward_hook(self.oDepthDecay)
        self.i_depth = 0
        self.f_depth = 0
        self.g_depth = 0
        self.o_depth = 0
        self.decay_alpha = 0.97
        self.igrad_weight = torch.zeros(hidden_size, input_size + hidden_size).cuda()
        self.fgrad_weight = torch.zeros(hidden_size, input_size + hidden_size).cuda()
        self.ggrad_weight = torch.zeros(hidden_size, input_size + hidden_size).cuda()
        self.ograd_weight = torch.zeros(hidden_size, input_size + hidden_size).cuda()
        self.igrad_bias = torch.zeros(hidden_size).cuda()
        self.fgrad_bias = torch.zeros(hidden_size).cuda()
        self.ggrad_bias = torch.zeros(hidden_size).cuda()
        self.ograd_bias = torch.zeros(hidden_size).cuda()

    def forward(self, input, hidden, context):
        self.i_depth = 0
        self.f_depth = 0
        self.g_depth = 0
        self.o_depth = 0
        self.igrad_weight = torch.zeros(self.hidden_size, self.input_size + self.hidden_size).cuda()
        self.fgrad_weight = torch.zeros(self.hidden_size, self.input_size + self.hidden_size).cuda()
        self.ggrad_weight = torch.zeros(self.hidden_size, self.input_size + self.hidden_size).cuda()
        self.ograd_weight = torch.zeros(self.hidden_size, self.input_size + self.hidden_size).cuda()
        self.igrad_bias = torch.zeros(self.hidden_size).cuda()
        self.fgrad_bias = torch.zeros(self.hidden_size).cuda()
        self.ggrad_bias = torch.zeros(self.hidden_size).cuda()
        self.ograd_bias = torch.zeros(self.hidden_size).cuda()
        combined = torch.cat((input, hidden), 1)
        i = F.sigmoid(self.igate(combined))
        g = F.tanh(self.ggate(combined))
        f = F.sigmoid(self.fgate(combined))
        o = F.sigmoid(self.ogate(combined))
        context_new = f*context + i*g
        hidden_new = o*F.tanh(context_new)

        return hidden_new, context_new

    def initHidden(self, bsz):
        return Variable(torch.zeros(bsz, self.hidden_size)).cuda()

    def iDepthDecay(self, module, grad_input, grad_output):
        self.i_depth += 1
        self.igrad_weight += grad_input[2].data.t() * (self.decay_alpha**self.i_depth)
        self.igrad_bias += grad_input[0].data.sum(0) * (self.decay_alpha**self.i_depth)
        return grad_input

    def fDepthDecay(self, module, grad_input, grad_output):
        self.f_depth += 1
        self.fgrad_weight += grad_input[2].data.t() * (self.decay_alpha**self.f_depth)
        self.fgrad_bias += grad_input[0].data.sum(0) * (self.decay_alpha**self.f_depth)
        return grad_input

    def gDepthDecay(self, module, grad_input, grad_output):
        self.g_depth += 1
        self.ggrad_weight += grad_input[2].data.t() * (self.decay_alpha**self.g_depth)
        self.ggrad_bias += grad_input[0].data.sum(0) * (self.decay_alpha**self.g_depth)
        return grad_input

    def oDepthDecay(self, module, grad_input, grad_output):
        self.o_depth += 1
        self.ograd_weight += grad_input[2].data.t() * (self.decay_alpha**self.o_depth)
        self.ograd_bias += grad_input[0].data.sum(0) * (self.decay_alpha**self.o_depth)
        return grad_input