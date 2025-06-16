from torch.distributions import RelaxedOneHotCategorical, RelaxedBernoulli
  #torch.distributions 是 PyTorch 中处理概率分布的模块。
  # 这里导入了 RelaxedOneHotCategorical 和 RelaxedBernoulli，但后者在代码中被使用，前者未被使用。
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
