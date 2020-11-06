import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from torch.autograd import Variable

from utils import *

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

class PGDAttack:

    def __init__(self, model, epsilon, num_steps, step_size, rand):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = rand
    
    def perturb_inf(self, x_nat, y, rand):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""

        x_nat = x_nat.cpu().numpy()
        y = y.cpu().numpy()

        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape).astype('float32')
            x = np.climp(x, 0, 1) # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        for i in range(self.num_steps):

            x_var = to_var(torch.from_numpy(x), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(x_var)
            loss = F.cross_entropy(scores, y_var)
            loss.backward()
            grad = x_var.grad.data.cpu().numpy()

            x += self.step_size * np.sign(grad)

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 1) # ensure valid pixel range

        return torch.from_numpy(x).cuda()

    def perturb_l2(self, x_nat, y, rand):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_2 norm."""
        if self.rand:
            pert = np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            pert_norm = np.linalg.norm(pert)
            pert = pert / max(1, pert_norm)
        else:
            pert = np.zeros(x_nat.shape)

        for i in range(self.num_steps):
            x = x_nat.cpu().numpy() + pert
            x = np.clip(x, 0, 1)
            x = to_var(torch.from_numpy(x).float(), requires_grad=True)

            scores = self.model(x)
            loss = F.cross_entropy(scores, y)
            loss.backward()
            grad = x.grad.data.cpu().numpy()

            normalized_grad = grad / np.linalg.norm(grad)
            pert = np.add(pert, self.step_size * normalized_grad, out=pert, casting='unsafe')

            # project pert to norm ball
            pert_norm = np.linalg.norm(pert)
            rescale_factor = pert_norm / self.epsilon
            pert = pert / max(1, rescale_factor)

        x = x_nat.cpu().numpy() + pert
        x = np.clip(x, 0, 1)

        return torch.from_numpy(x).float().cuda()


