from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RoPECache, apply_rope_single
from kv_cache import KVCache  

#GQA: Grouped Query Attention (MHA, MQA, GQA)
# dưới đây là GQA hiện đại hơn MHA truyền thống
class CausalSelfAttentionModern(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0,
                 n_kv_head: int | None = None):  # ← NEW
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head # số head của query
        self.n_kv_head = n_kv_head or n_head #số head của k và v      # có thể là MHA hoac GQA
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        self.group_size = self.n_head // self.n_kv_head # số head query trên mỗi head kv
        self.d_head = n_embd // n_head

        # Separate projections for Q vs K/V (sizes differ under GQA)  ← CHANGED
        self.wq  = nn.Linear(n_embd, self.n_head   * self.d_head, bias=False)
        self.wk  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.use_rope = rope
        self.rope_cache: RoPECache | None = None
        self.max_pos = max_pos # maximum position for RoPE
        self.sliding_window = sliding_window #kích thước cửa sổ trượt
        self.attention_sink = attention_sink #số token đầu tiên được giữ lại khi dùng sliding window

    def _maybe_init_rope(self, device):
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pos, device=device)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        '''
        Hàm forward này xử lý cả hai luồng huấn luyện (training) và suy luận (inference).

        --- Luồng Huấn luyện (Training / Prefill) ---
        (Khi `kv_cache` là `None`)
        1.  Input `x` có kích thước (B, T, C), với T là độ dài đầy đủ của chuỗi.
        2.  `x` được chiếu (project) thành:
            * `q`: (B, T, n_head, d_head)
            * `k`: (B, T, n_kv_head, d_head)
            * `v`: (B, T, n_kv_head, d_head)
        3.  `q` và `k` được áp dụng RoPE (Rotary Positional Embedding) cho toàn bộ chuỗi (từ 0 đến T-1).
        4.  (Nếu là GQA/MQA): Mỗi head K/V (`n_kv_head`) được lặp lại `group_size` lần (dùng `repeat_interleave`) để số lượng head K/V khớp với số head Q (`n_head`).
        5.  Tính toán chú ý (F.scaled_dot_product_attention) với `is_causal=True` để tạo mặt nạ (mask) tự hồi quy, đảm bảo token ở vị trí `i` chỉ chú ý đến các token từ `0` đến `i`.
        6.  Kết quả `y` được trả về 

        --- Luồng Suy luận (Inference / Decoding) ---
        (Khi `kv_cache` *không* phải `None`)
        1.  Lúc này, `x` chỉ chứa 1 token mới, nên `T=1`. `start_pos` là vị trí của token này (ví dụ: 5).
        2.  `x` (B, 1, C) được chiếu thành `q`, `k`, `v` (cũng có T=1).
        3.  Chỉ `q` và `k` *mới* này được áp dụng RoPE tại vị trí `start_pos` (ví dụ: vị trí 5).
        4.  Lấy K/V của các token quá khứ (đã được áp dụng RoPE) từ `kv_cache` (ví dụ: từ 0 đến 4).
        5.  Nối K *mới* (token 5) vào K *cũ* (token 0-4) để tạo `k_all` (B, Hk, T_past + 1, D). Tương tự cho `v_all`.
        6.  (Nếu là SWA): Cắt bớt `k_all`, `v_all` để chỉ giữ lại các token trong cửa sổ trượt (sliding window) và các "attention sinks".
        7.  (Nếu là GQA/MQA): Lặp lại `k_all` và `v_all` (đã bao gồm cả quá khứ) để khớp với số head Q.
        8.  Tính chú ý: `q` (B, H, 1, D) sẽ chú ý đến `k_attn` (B, H, T_past + 1, D).
            * `is_causal` được đặt là `False`. Điều này là an toàn và cần thiết, vì `q` chỉ có 1 token ở vị trí cuối cùng, nó *phải* được phép nhìn thấy *tất cả* các key trong `k_attn` (toàn bộ quá khứ).
        9.  Cập nhật `kv_cache`: Lưu K/V *mới* (dạng *gọn* `Hk`, không phải dạng đã lặp lại) vào cache.
        10. Trả về `y` (kích thước B, 1, C) và `KVCache` đã được cập nhật.
        '''
        B, T, C = x.shape
        self._maybe_init_rope(x.device)

        # Projections
        q = self.wq(x).view(B, T, self.n_head,    self.d_head).transpose(1, 2)    # (B, H,  T, D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B, Hk, T, D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B, Hk, T, D)

        # RoPE on *current* tokens (cached keys are already rotated)
        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cos, sin = self.rope_cache.get(pos)
            q = apply_rope_single(q, cos, sin)   # (B, H,  T, D)
            k = apply_rope_single(k, cos, sin)   # (B, Hk, T, D)

        # Concatenate past cache (cache is stored in Hk heads)
        if kv_cache is not None:
            # kv_cache.k/v có shape (B, Hk, T_past, D)
            # k/v mới có shape (B, Hk, T, D) - với T=1 khi inference
            k_all = torch.cat([kv_cache.k, k], dim=2)   # (B, Hk, T_past + T, D)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            # Đây là luồng training hoặc prefill
            k_all, v_all = k, v

        # Sliding-window + attention-sink (crop along seq length)
        # Cắt bớt K/V trong cache nếu vượt quá cửa sổ trượt
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            s = self.attention_sink
            # Giữ lại 's' token đầu tiên (sinks) và 'sliding_window' token cuối cùng
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)

        # --- GQA expand: repeat K/V heads to match Q heads before attention ---
        if self.n_kv_head != self.n_head: # Nếu là GQA hoặc MQA
            # k_all từ (B, Hk, Tk, D) -> (B, H, Tk, D)
            k_attn = k_all.repeat_interleave(self.group_size, dim=1) # lặp lại key cho mỗi head query
            v_attn = v_all.repeat_interleave(self.group_size, dim=1) # lặp lại value cho mỗi head query
        else:
            # Nếu là MHA (n_kv_head == n_head)
            k_attn, v_attn = k_all, v_all # không cần lặp lại

        # Scaled dot-product attention (PyTorch scales internally)
        # is_causal = True chỉ khi đang training/prefill (kv_cache=None)
        # is_causal = False khi đang inference (kv_cache != None)
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(q, k_attn, v_attn,
                                               attn_mask=None,
                                               dropout_p=self.dropout.p if self.training else 0.0,
                                               is_causal=is_causal)         # (B, H, T, D)

        # Reshape output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y) # Chiếu (project) đầu ra cuối cùng

        # Update KV cache (store compact Hk heads, not expanded)
        if kv_cache is not None:
            # Chỉ cập nhật cache khi đang ở chế độ inference
            # k_all và v_all đã chứa cache cũ + token mới
            new_cache = KVCache(k_all, v_all)
        else:
            # Khi training/prefill, tạo cache mới từ đầu
            new_cache = KVCache(k, v)
            
        return y, new_cache