from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class KVCache:
    k: torch.Tensor  # (B,H,T,D) #lưu key
    v: torch.Tensor  # (B,H,T,D) #lưu value

    @property
    def T(self):
        return self.k.size(2) #số lượng token đã lưu trong cache

class RollingKV:
    """Rolling buffer with optional attention sink.
    Keeps first `sink` tokens + last `window` tokens.
    """
    def __init__(self, window: int, sink: int = 0):
        self.window = window #số token cuối cùng được giữ lại
        self.sink = sink #số token đầu tiên được giữ lại
        self.k = None 
        self.v = None
    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None: #nếu chưa có cache thì khởi tạo
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2) #nối key mới vào cache
            self.v = torch.cat([self.v, v_new], dim=2) #nối value mới vào cache
        # crop
        if self.k.size(2) > self.window + self.sink: #nếu số token trong cache vượt quá giới hạn
            sink_part = self.k[:, :, :self.sink, :] #phần token đầu tiên được giữ lại
            sink_val  = self.v[:, :, :self.sink, :] #phần value tương ứng
            tail_k = self.k[:, :, -self.window:, :] #phần token cuối cùng được giữ lại
            tail_v = self.v[:, :, -self.window:, :] #phần value tương ứng
            self.k = torch.cat([sink_part, tail_k], dim=2) #cập nhật lại cache key
            self.v = torch.cat([sink_val, tail_v], dim=2) #cập nhật lại cache value
        return self.k, self.v