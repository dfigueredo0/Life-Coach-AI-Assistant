from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset

class RMPairs(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer, max_len: int = 512):
        self.path = Path(jsonl_path)
        self.rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        r = self.rows[i]
        q = r.get("prompt", "")
        pos = r.get("chosen", "")
        neg = r.get("rejected", "")

        pos_enc = self.tok(q, pos, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        neg_enc = self.tok(q, neg, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

        return {
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
        }

def collate_rm(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
