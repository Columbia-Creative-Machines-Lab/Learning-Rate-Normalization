from torch.optim import Optimizer
from torch import nn
import numpy as np
import math

class SGD_lr_norm(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = SGD_lr_norm(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD_lr_norm with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in constrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, schedule=None, gamma=0.1,
                 hidden_depth=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_lr_norm, self).__init__(params, defaults)
        self.schedule = schedule
        self.gamma = gamma
        self._hidden_depth = 0
        self._hidden_depth_ref = hidden_depth

    def __setstate__(self, state):
        super(SGD_lr_norm, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # Was thinking of using this as a method for incrementing based on hidden but it seems impractical
        if self._hidden_depth_ref is not None:
            self._hidden_depth = self.__get_hidden_depth()
        loss = None
        if closure is not None:
            loss = closure()
        global_step = 0.0
        decay_steps = 1.0
        i = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # -2 for the last layer of weights, -1 is the output activations
            w_mul = np.linalg.norm(group['params'][-2].grad.data.cpu().numpy())
            b_mul = 1
            avg_norm = 0.0
            i = 0.0
            #for j in range(1, len(group['params'])/2):
            #    weight = group['params'][-2*j].grad.data.cpu().numpy()
            #    weight_norm = np.linalg.norm(weight)
            #    avg_norm += weight_norm
            #    i += 1.0
            #    #if weight_norm > max_norm: min_norm = weight_norm
            #w_mul = avg_norm/i

            #w_mul = 1
            # d is for depth
            d = 0
            #group['lr'] *= 1-1e-4
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # Can decay learning rate here
                # seems like you can get layer information from the p iteration and the group iteration
                # can write a function here that takes in a schedule and decays according to that

                # TODO: write learning rate annealing (scheduler) lower learning rate after epochs

                #if self.schedule != 'none':

                # This conditional prevents the optimizer from considering things such as batch-norm/dropout
                # as if they were extra depth to the network
                if p.data.shape[0] > 1 and self.schedule != 'none':
                    if self.schedule == 'exponential':
                        norm = np.linalg.norm(d_p.cpu().numpy())
                        if norm == 0:
                            p.data.add_(-group['lr'], d_p)
                        else:
                            new_lr = -group['lr']/norm
                            new_lr *= w_mul
                            new_lr = new_lr * math.exp(-self.gamma*d)
                            p.data.add_(new_lr, d_p)
                    elif self.schedule == 'linear':
                        norm = np.linalg.norm(d_p.cpu().numpy())
                        if norm == 0:
                            p.data.add_(-group['lr'], d_p)
                        else:
                            new_lr = -group['lr']/norm
                            new_lr *= w_mul
                            new_lr -= new_lr*((1-self.gamma)**d)
                            p.data.add_(new_lr, d_p)
                    else: # Normalize
                        norm = np.linalg.norm(d_p.cpu().numpy())
                        if norm == 0:
                            p.data.add_(-group['lr'], d_p)
                        else:
                            new_lr = -group['lr']/norm
                            new_lr *= w_mul
                            p.data.add_(new_lr, d_p)
                elif p.data.shape[0] == 1 and self.schedule != 'none':
                    if self.schedule == 'exponential':
                        norm = np.linalg.norm(d_p.cpu().numpy())
                        if norm == 0:
                            p.data.add_(-group['lr'], d_p)
                        else:
                            new_lr = -group['lr']/norm
                            new_lr *= b_mul
                            new_lr = new_lr * math.exp(-self.gamma*d)
                            p.data.add_(new_lr, d_p)
                    elif self.schedule == 'linear':
                        norm = np.linalg.norm(d_p.cpu().numpy())
                        if norm == 0:
                            p.data.add_(-group['lr'], d_p)
                        else:
                            new_lr = -group['lr']/norm
                            new_lr *= b_mul
                            new_lr -= new_lr*((1-self.gamma)**d)
                            p.data.add_(new_lr, d_p)
                    else: # Normalize
                        norm = np.linalg.norm(d_p.cpu().numpy())
                        if norm == 0:
                            p.data.add_(-group['lr'], d_p)
                        else:
                            new_lr = -group['lr']/norm
                            new_lr *= b_mul
                            p.data.add_(new_lr, d_p)
                else:
                    p.data.add_(-group['lr'], d_p)
                if p.data.shape[0] > 1:
                    d += 1
            i += 1

        return loss

    def __get_hidden_depth(self):
        return self._hidden_depth_ref()

    def inc_hidden(self):
        self._hidden_depth += 1
