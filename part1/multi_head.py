from torch import nn
import torch
from attn_mask import causal_mask

class MyMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, trace_shapes: bool = True):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model) # W_Q, W_K, W_V
        self.out_proj = nn.Linear(d_model, d_model) # W_O
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x):
        B, T, _ = x.shape  # batch size, sequence length, model dimension
        qkv = self.qkv(x)
        if self.trace_shapes:
            print(f"After qkv linear: {x.shape}")
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each is (B,T,n_head,d_head)
        q = q.transpose(1, 2)  # (B,n_head,T,d_head)
        k = k.transpose(1, 2)  # (B,n_head,T,d_head)
        v = v.transpose(1, 2)  # (B,n_head,T,d_head)
        if self.trace_shapes:
            print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B,n_head,T,T)
        mask = causal_mask(T, device=x.device)  # (1,1,T,T)
        scores = scores.masked_fill(mask, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)  # (B,n_head,T,T)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)  # (B,n_head,T,d_head)
        if self.trace_shapes:
            print(f"Context before merge: {context.shape}")
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B,T,d_model)
        out = self.out_proj(context)  # (B,T,d_model)
        if self.trace_shapes:
            print(f"Output after out_proj: {out.shape}")
        return out, weights
