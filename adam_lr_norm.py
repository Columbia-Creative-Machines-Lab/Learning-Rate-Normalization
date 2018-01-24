import math
import torch
import numpy as np
from torch.optim import Optimizer


class Adam_lr_norm(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, schedule=None, gamma=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(Adam_lr_norm, self).__init__(params, defaults)
        self.schedule = schedule
        self.gamma = gamma

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        d = 0
        weight_norms = []
        bias_norms = []


        for group in self.param_groups:
            #w_mul = np.linalg.norm(self.param_groups[-2].grad.data.cpu().numpy())
            w_mul = 1.0
            b_mul = 1.0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # normalization
                new_grad = grad
                norm = torch.norm(grad)
                if len(p.data.shape) > 1:
                    if self.schedule == 'exponential':
                        if norm != 0:
                            new_grad = w_mul*new_grad/norm
                            new_grad = new_grad * math.exp(-self.gamma*d)
                    elif self.schedule == 'linear':
                        if norm != 0:
                            new_grad = w_mul*new_grad/norm
                            new_grad = new_grad * ((1-self.gamma)**d)
                    elif self.schedule != 'none':
                        if norm != 0:
                            new_grad = w_mul*new_grad/norm
                    weight_norms.append(norm)
                elif len(p.data.shape) == 1:
                    if self.schedule == 'exponential':
                        if norm != 0:
                            new_grad = b_mul*new_grad/norm
                            new_grad = new_grad * math.exp(-self.gamma*d)
                    elif self.schedule == 'linear':
                        if norm != 0:
                            new_grad = b_mul*new_grad/norm
                            new_grad = new_grad * ((1-self.gamma)**d)
                    elif self.schedule != 'none':
                        if norm != 0:
                            new_grad = b_mul*new_grad/norm
                    bias_norms.append(norm)
                if p.data.shape[0] > 1:
                    d += 1

                grad = new_grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss, [weight_norms, bias_norms]
