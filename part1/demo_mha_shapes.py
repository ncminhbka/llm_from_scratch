"""Walkthrough of multi-head attention with explicit matrix math and shapes.
Generates a text log at ./out/mha_shapes.txt.
"""
import os
import math
import torch
from multi_head import MyMultiheadAttention

OUT_TXT = os.path.join(os.path.dirname(__file__), "out", "mha_shapes.txt")

def log(msg: str):
    print(msg)
    with open(OUT_TXT, "a") as f:
        f.write(msg + "\n")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    open(OUT_TXT, "w").close()  # clear previous log

    B, T, d_model, n_head = 1, 5, 12, 3 # batch size, sequence length, model dim, number of heads
    d_head = d_model // n_head

    x = torch.randn(B, T, d_model) #input tensor
    attn = MyMultiheadAttention(d_model=d_model, n_head=n_head)

    log(f"Input x:           {tuple(x.shape)} = (B,T,d_model)")
    qkv = attn.qkv(x)  # (B,T,3*d_model) 
    log(f"Linear qkv(x):     {tuple(qkv.shape)} = (B,T,3*d_model)")

    qkv = qkv.view(B, T, 3, n_head, d_head)  # (B,T,3,n_head,d_head)
    log(f"Reshape qkv:       {tuple((B, T, 3, n_head, d_head))} = (B,T,3,n_head,d_head)")
    q, k, v = qkv.unbind(dim=2)  # each is (B,T,n_head,d_head)
    log(f"Unbind q,k,v:      {tuple(q.shape)} = (B,T,n_head,d_head) each")

    q = q.transpose(1, 2)  # (B,n_head,T,d_head)
    k = k.transpose(1, 2)  # (B,n_head,T,d_head)
    v = v.transpose(1, 2)  # (B,n_head,T,d_head)
    log(f"Transpose q,k,v:   {tuple(q.shape)} = (B,n_head,T,d_head) each")

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)  # (B,n_head,T,T)
    log(f"scores q@k^T:      {tuple(scores.shape)} = (B,heads,T,T)")

    weights = torch.softmax(scores, dim=-1)  # (B,n_head,T,T)
    log(f"softmax(scores):   {tuple(weights.shape)} = (B,heads,T,T)")

    out = torch.matmul(weights, v)  # (B,n_head,T,d_head)
    log(f"weights@v:        {tuple(out.shape)} = (B,heads,T,d_head)")

    out = out.transpose(1, 2).contiguous().view(B, T, d_model)  # (B,T,d_model)
    log(f"merge heads:       {tuple(out.shape)} = (B,T,d_model)")

    out = attn.out_proj(out)  # (B,T,d_model) out = context @ W_O
    log(f"out_proj:          {tuple(out.shape)} = (B,T,d_model)")
    