import torch
import torch.nn as nn
import torch.nn.functional as F

class SIGRegLoss(nn.Module):
    def __init__(self, input_dim, proj_dim=128):
        super(SIGRegLoss, self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        
        # Initialize random projection matrix A
        A = torch.randn(input_dim, proj_dim)
        # CRITICAL: Normalize the random projection matrix A so columns have norm=1
        self.A = nn.Parameter(A / A.norm(dim=0), requires_grad=False)

    def forward(self, x):
        # Projects input x using the normalized matrix A and computes a regularization loss
        # (Example: encouraging projected representations to be sparse or distributed)
        z = x @ self.A
        loss = torch.mean(torch.abs(z)) # L1 regularization on projections as an example
        return loss

class VectorFlow(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super(VectorFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        return self.net(x)