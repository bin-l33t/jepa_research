# AGENT: ARCHITECT + MATH_CONTEXT
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorFlow(nn.Module):
    """
    A simple MLP predictor (VectorFlow).
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=None):
        super(VectorFlow, self).__init__()
        if output_dim is None:
            output_dim = input_dim
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SIGRegLoss(nn.Module):
    """
    SIGRegLoss: Regularization using a random projection matrix.
    Follows Math Specs: A must be normalized so columns have norm=1.
    """
    def __init__(self, input_dim, proj_dim=64):
        super(SIGRegLoss, self).__init__()
        # Random projection matrix A (fixed, not learnable)
        self.register_buffer('A', torch.randn(input_dim, proj_dim))

    def forward(self, z):
        # CRITICAL MATH REQUIREMENT: Normalize A so columns have norm=1
        # Formula: A = A / A.norm(dim=0)
        A_norm = self.A / (self.A.norm(dim=0, keepdim=True) + 1e-8)
        
        # Project input Z onto A
        # z: [Batch, Input_Dim]
        # A_norm: [Input_Dim, Proj_Dim]
        # p: [Batch, Proj_Dim]
        p = torch.mm(z, A_norm)
        
        # Calculate variance and covariance of projections
        # Center the projections
        p_centered = p - p.mean(dim=0, keepdim=True)
        batch_size = z.size(0)
        
        # Covariance matrix of projections: (P.T @ P) / (N - 1)
        cov_p = torch.mm(p_centered.t(), p_centered) / (batch_size - 1 + 1e-8)
        
        # Regularization:
        # 1. Variance Term: Encourage variance of each projected dimension to be at least 1
        #    using a hinge loss: mean(relu(1 - std))
        std_p = torch.sqrt(torch.diagonal(cov_p) + 1e-8)
        var_loss = torch.mean(torch.relu(1 - std_p))
        
        # 2. Covariance Term: Minimize off-diagonal correlations
        #    Sum of squared off-diagonal elements
        off_diag_mask = ~torch.eye(cov_p.size(0), device=cov_p.device).bool()
        cov_loss = torch.sum(cov_p[off_diag_mask] ** 2) / cov_p.size(0)
        
        return var_loss + cov_loss
---------------------------------