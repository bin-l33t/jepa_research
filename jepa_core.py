# AGENT: ARCHITECT + MATH_CONTEXT

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
class VectorFlow(nn.Module):
    """
    A simple MLP predictor as specified in architect.txt.
    """
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super(VectorFlow, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        if out_dim is None:
            out_dim = in_dim
            
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class SIGRegLoss(nn.Module):
    """
    SIGRegLoss implementation following Math Specs.
    Uses a random projection matrix A to enforce isotropy in the embedding space.
    """
    def __init__(self, in_dim, proj_dim=128):
        super(SIGRegLoss, self).__init__()
        
        # Initialize random projection matrix A
        # We register it as a buffer so it persists but is not a learnable parameter
        self.register_buffer('A', torch.randn(in_dim, proj_dim))
        
        # CRITICAL MATH REQUIREMENT:
        # Normalize the random projection matrix A so columns have norm=1.
        # Formula: A = A / A.norm(dim=0)
        self.A.div_(self.A.norm(dim=0, keepdim=True))

    def forward(self, z):
        """
        Computes the regularization loss.
        
        Args:
            z (torch.Tensor): Input embeddings of shape [Batch, Dim]
            
        Returns:
            torch.Tensor: Scalar loss value (Variance + Covariance regularization on projections)
        """
        batch_size = z.size(0)
        
        # Project embeddings: P = Z @ A
        # z: [B, D], A: [D, K] -> p: [B, K]
        p = torch.matmul(z, self.A)
        
        # Center the projections along the batch dimension
        p = p - p.mean(dim=0)
        
        # Compute Covariance Matrix of the projected embeddings
        # cov: [K, K]
        cov = torch.matmul(p.T, p) / (batch_size - 1 + 1e-6)
        
        # Regularization:
        # 1. Variance Term: Encourage diagonal elements to be 1 (maintain information/variance)
        # Using MSE on variance: mean((std - 1)^2) equivalent to mean((var - 1)^2) roughly for this purpose
        var_loss = torch.mean((torch.diagonal(cov) - 1).pow(2))
        
        # 2. Covariance Term: Encourage off-diagonal elements to be 0 (decorrelation)
        num_proj = self.A.shape[1]
        # Create a mask for off-diagonal elements
        off_diag_mask = ~torch.eye(num_proj, device=cov.device).bool()
        cov_loss = torch.mean(cov[off_diag_mask].pow(2))
        
        # Total regularization loss
        return var_loss + cov_loss