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
    
def _stream_jsonl(path: Path) -> Iterator[Dict]:
    with(open(path, 'r', encoding='utf-8')) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _format_turns(rec: Dict, sp: Dict[str, str]) -> str:
    buf = []
    for t in rec['turns']:
        if t['role'] == 'user':
            buf.append(f"{sp['user']}{t['text']}{sp['end']}")
        elif t['role'] == 'assistant':
            buf.append(f"{sp['assistant']}{t['text']}{sp['end']}")
        elif t['role'] == 'system':
            buf.append(f"{sp['system']}{t['text']}{sp['end']}")
    return ''.join(buf)

class SFTJsonlDataset(Dataset):
    def __init__(self, file: Path, tokenizer: PreTrainedTokenizerBase, special: Dict[str, str], max_len: int):
        self.tokenizer = tokenizer
        self.special = special
        self.max_len = max_len
        self.examples: List[DialogExample] = []
        for rec in _stream_jsonl(file):
            self.examples.append(DialogExample(text=_format_turns(rec=rec, sp=special)))
            
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        text = self.examples[idx].text
        tokens = self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        tokens['labels'] = tokens['input_ids'].clone()
        return {k: v.squeeze(0) for k, v in tokens.items()}