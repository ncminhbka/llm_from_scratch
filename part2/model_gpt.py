from __future__ import annotations
import torch
from torch import nn
import math
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)  # (B, T, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)  # each is (B, T, num_heads, head_dim)
        
        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)  # (B, num_heads, T, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, T, head_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)
        
        mask = torch.tril(torch.ones(T, T)).to(x.device)  # (T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, T, T)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        out = self.out_proj(context)  # (B, T, C)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.GELU(),
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout),   
    )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.vocab_size = vocab_size

        self.apply(self._init_weights) # áp dụng cho mọi layer như Linear, Embedding, LayerNorm

    def _init_weights(self, module):
        if(isinstance(module, nn.Linear)): #nếu là Linear
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding): #nếu là Embedding
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.block_size, "Sequence length exceeds block size"

        token_embeddings = self.token_emb(idx)  # (B, T, n_embd)
        position_ids = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        position_embeddings = self.pos_emb(position_ids)  # (1, T, n_embd)

        x = token_embeddings + position_embeddings  # (B, T, n_embd)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)  # (B, T, n_embd)
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
        

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0,
                top_k: int | None = 50, top_p: float | None = None):
        from utils import top_k_top_p_filtering
        self.eval()
        # Guard: if the prompt is empty, start with a newline byte (10)
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] # crop to block size (context length)
            logits, _ = self(idx_cond) # (B, T, vocab_size)
            logits = logits[:, -1, :] / max(temperature, 1e-6) #logits dự đoán token tiếp theo chính là logits của token cuối cùng
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1) # lấy ngẫu nhiên theo phân phối
            idx = torch.cat([idx, next_id], dim=1)
        return idx # (B, T + max_new_tokens)