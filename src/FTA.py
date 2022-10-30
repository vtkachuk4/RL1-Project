import torch as T
import torch.nn as nn

# Taken from labml.ai: https://nn.labml.ai/activations/fta/index.html 

class FTA(nn.Module):
    def __init__(self, lower_limit: float, upper_limit: float, delta: float, eta: float):
        super().__init__()
        self.c = nn.Parameter(T.arange(lower_limit, upper_limit, delta), requires_grad = False)
        self.expansion_factor = len(self.c)
        self.delta = delta
        self.eta = eta

    def fuzzy_i_plus(self, x: T.Tensor):
        return (x <= self.eta) * x + (x > self.eta)

    def forward(self, z: T.tensor):
        # add another dimension of size 1 to expand into bins. *T.shape unpacks list into individual ints: T.view(T.shape, 1) = T.view((N, C), 1) is incorrect; T.view(*T.shape, 1) = T.view(N, C, 1) is correct
        z = z.view(*z.shape, 1)
        # fuzzy tiling activation function (clip clamps all elements in input into range [min, max])
        z = 1.0 - self.fuzzy_i_plus(T.clip(self.c - z, min=0.0) + T.clip(z - self.delta - self.c, min=0.0))
        # reshape back to original number of dimensions. Last dim size gets expanded by number of bins, (upper-lower)/delta
        return z.view(*z.shape[:-2], -1)
    
