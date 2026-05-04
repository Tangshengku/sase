import torch
import torch.nn as nn


class SAESSVDLinear(nn.Module):
    def __init__(self, a_weight, b_weight, bias=None):
        super().__init__()
        out_features, rank = a_weight.shape
        rank_b, in_features = b_weight.shape
        if rank != rank_b:
            raise ValueError(f"incompatible factor shapes: {a_weight.shape} and {b_weight.shape}")
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias is not None)
        self.truncation_rank = rank
        self.ALinear.weight.data.copy_(a_weight)
        self.BLinear.weight.data.copy_(b_weight)
        if bias is not None:
            self.ALinear.bias.data.copy_(bias)

    def forward(self, x):
        return self.ALinear(self.BLinear(x))

