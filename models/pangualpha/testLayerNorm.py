import torch

class LayerNorm_(torch.nn.Module):
    def __init__(self, normalized_shape, init_method, dp=4, eps=1e-5, scale=1e-3):
        super(LayerNorm_, self).__init__()
        # Initialize master weight
        self.gamma = torch.empty(normalized_shape,
                                 dtype=torch.float,
                                 requires_grad=True)
        init_method(self.gamma)
        self.beta = torch.empty(normalized_shape,
                                dtype=torch.float,
                                requires_grad=True)
        init_method(self.beta)
        self.eps = eps

    def construct(self, x):
        mean = torch.mean(x, -1, keepdim=True)
        diff = torch.sub(x, mean)
        variance = torch.mean(torch.square(diff), -1,keepdim=True)
        variance_eps = torch.sqrt(torch.add(variance, self.eps))
        output_ = torch.div(diff, variance_eps)
        output = torch.add(torch.mul(output_, self.gamma), self.beta)
        return output
