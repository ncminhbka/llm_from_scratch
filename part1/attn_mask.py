import torch

def causal_mask(T: int, device=None):
    """Returns a bool mask where True means *masked* (disallowed).
    Shape: (1, 1, T, T) suitable for broadcasting with (B, heads, T, T) (scores).
    """
    m = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
    return m.view(1, 1, T, T)
#this mask will be added to the attention scores (T, T) before softmax to prevent attending to future tokens