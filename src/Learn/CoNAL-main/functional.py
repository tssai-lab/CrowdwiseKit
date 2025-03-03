from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli
import torch.nn as nn


def gumbel_sigmoid(input, temp):
    return RelaxedBernoulli(temp, probs=input).rsample()


class GumbelSigmoid(nn.Module):
    def __init__(self,
                 temp: float = 0.1,
                 threshold: float = 0.5):
        super(GumbelSigmoid, self).__init__()
        self.temp = temp
        self.threshold = threshold

    def forward(self, input):
        if self.training:
            return gumbel_sigmoid(input, self.temp)
        else:
            return (input.sigmoid() >= self.threshold).float()