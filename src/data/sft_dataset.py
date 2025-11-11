from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

@dataclass
class SFTJsonlDataset(Dataset):
    file: Path
    tokenizer: PreTrainedTokenizerBase
    special: Dict[str, str]
    max_len: int

    def __init__(self, file: Path, tokenizer: PreTrainedTokenizerBase, special: Dict[str, str], max_len: int):
        self.items: List[Dict] = []
        self.tokenizer = tokenizer
        self.special = special
        self.max_len = max_len
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))

    def _linearize_turns(self, ex: Dict) -> str:
        usr = self.special.get("user", "<|user|>")
        sys = self.special.get("system", "<|system|>")
        asst = self.special.get("assistant", "<|assistant|>")
        end = self.special.get("end", "<|endoftext|>")
        parts: List[str] = []
        for t in ex.get("turns", []):
            role = t.get("role", "")
            content = t.get("content", "")
            if role == "system":
                parts.append(f"{sys}{content}{end}")
            elif role == "user":
                parts.append(f"{usr}{content}{end}")
            elif role == "assistant":
                parts.append(f"{asst}{content}{end}")
            else:
                parts.append(content)
        return "".join(parts)

    def _format_fallback(self, ex: Dict) -> str:
        # If transforms wrote a packed "text" field, just use it.
        if isinstance(ex.get("text"), str):
            return ex["text"]
        # Otherwise, try to linearize
        if "turns" in ex:
            return self._linearize_turns(ex)
        # Last resort
        return str(ex)

    def __getitem__(self, idx):
        ex = self.items[idx]
        text = self._format_fallback(ex)
        return {"text": text}  # IMPORTANT: collator expects dict with 'text'

    def __len__(self):
        return len(self.items)
