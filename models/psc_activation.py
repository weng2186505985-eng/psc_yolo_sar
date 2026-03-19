import torch
import torch.nn as nn

class PSCActivation(nn.Module):
    def __init__(self, num_parameters):
        super().__init__()
        self.A     = nn.Parameter(torch.ones(num_parameters) * 1.0)
        self.x0    = nn.Parameter(torch.zeros(num_parameters))
        self.alpha = nn.Parameter(torch.ones(num_parameters) * 0.5)
        self.beta  = nn.Parameter(torch.ones(num_parameters) * 0.5)

    def forward(self, x):
        A     = torch.abs(self.A)
        alpha = torch.clamp(self.alpha, min=0.01, max=5.0)
        beta  = torch.clamp(self.beta,  min=0.01, max=5.0)
        dims  = [1] * x.dim()
        dims[1] = -1
        diff  = torch.abs(x - self.x0.view(*dims))
        return A.view(*dims) * torch.pow(diff + 1e-6, alpha.view(*dims)) \
                             * torch.exp(-beta.view(*dims) * diff)
