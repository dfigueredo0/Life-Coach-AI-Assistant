from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterator, List

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

@dataclass
class DialogExample:
    text: str
    
class SFTJsonlDataset(Dataset):
    def __init__(self, file: Path, tokenizer: PreTrainedTokenizerBase, special: Dict[str, str], max_len: int):
        self.path = Path(file)
        self.rows = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.max_len = max_len
        self.specials = special
        self.tokenizer = tokenizer
            
    def __len__(self):
        return len(self.rows)
    
    def _linearize_turns(self, turns):
        sys_tok = self.specials.get("system", "<|system|>")
        usr_tok = self.specials.get("user", "<|user|>")
        as_tok  = self.specials.get("assistant", "<|assistant|>")
        end_tok = self.specials.get("end", "<|endoftext|>")

        pieces = []
        for t in turns:
            role = t.get("role","")
            content = t.get("content","")
            if role == "system":    pieces.append(f"{sys_tok}{content}{end_tok}")
            elif role == "user":    pieces.append(f"{usr_tok}{content}{end_tok}")
            elif role == "assistant": pieces.append(f"{as_tok}{content}{end_tok}")
            else:                   pieces.append(content)
        return "".join(pieces)

    def __getitem__(self, idx):
        r = self.rows[idx]
        if isinstance(r.get("text"), str):
            text = r["text"]
        elif "turns" in r:
            text = self._linearize_turns(r["turns"])
        else:
            # last resort: stringify
            text = str(r)
        return {"text": text}