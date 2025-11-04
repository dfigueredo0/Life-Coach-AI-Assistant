from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollator:
    """
    Collates JSONL records like:
      {"text": "<|system|>...<|endoftext|><|user|>...<|endoftext|><|assistant|>...<|endoftext|>"}
    Tokenizes, pads, and masks USER spans (labels=-100) so loss trains only on assistant/system if desired.

    Args:
        tokenizer: HF tokenizer with the four special tokens already added:
                   <|system|>, <|user|>, <|assistant|>, <|endoftext|>.
        max_length: int max seq length (pads/truncates to this).
        mask_user: if True, sets labels=-100 over [<|user|> ... <|endoftext|>] segments.
                   Assistant & system tokens remain trainable.
    """
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 2048
    mask_user: bool = True
    
    def __post_init__(self):
        self.tok_id_user = self.tokenizer.convert_tokens_to_ids("<|user|>")
        self.tok_id_assistant = self.tokenizer.convert_tokens_to_ids("<|assistant|>")
        self.tok_id_system = self.tokenizer.convert_tokens_to_ids("<|system|>")
        self.tok_id_end = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"
            
    def _mask_user_segments(self, input_ids: torch.Tensor, labels: torch.Tensor):
        ids = input_ids.tolist()
        n = len(ids)
        i = 0
        
        while i < n:
            if ids[i] == self.tok_id_user:
                j = i + 1
                while j < n and ids[j] != self.tok_id_end:
                    j += 1
                if j < n and ids[j] == self.tok_id_end:
                    labels[i : j + 1] = -100
                    i = j + 1
                    continue
                else:
                    labels[i:] = -100
                    break
            i += 1
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f.get('text') or f.get('rendered_text') for f in features]
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        input_ids = enc['inputs_ids']
        attention_mask = enc['attention_mask']
        labels = input_ids.clone()
        
        if self.mask_user:
            for b in range(input_ids.size(0)):
                self._mask_user_segments(input_ids[b], labels[b])
                
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        return batch