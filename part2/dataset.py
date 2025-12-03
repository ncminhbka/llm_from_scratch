from __future__ import annotations
import torch
from pathlib import Path

class ByteDataset:
    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        data = Path(path).read_bytes() #đọc data dưới dạng byte
        data = torch.tensor(list(data), dtype=torch.long) #chuyển đổi danh sách byte thành tensor long
        n = int(len(data) * split) #chia data thành train và val
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size #context length

    def get_batch(self, which: str, batch_size: int, device: torch.device):
        data = self.train if which == 'train' else self.val #chọn data train hoặc val
        ix = torch.randint(len(data) - self.block_size, (batch_size,)) # random start indices
        x = torch.stack([data[i:i + self.block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix]).to(device)
        return x, y
        