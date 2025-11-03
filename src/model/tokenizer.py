from __future__ import annotations

from dataclasses import dataclass
from transformers import AutoTokenizer

from typing import Dict, Any

SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|endoftext|>",
    ]
}

@dataclass
class TokenizerBundle:
    tokenizer: Any
    special: Dict[str, str]
    
def load_tokenizer(backbone_name: str, special_tokens: Dict[str, str]) -> TokenizerBundle:
    token = AutoTokenizer.from_pretrained(backbone_name, use_fast=True, padding_side="right", truncation_side="right")
    added = token.add_special_tokens(SPECIAL_TOKENS)
    if token.eos_token is None:
        token.eos_token = "<|endoftext|>"
    if token.pad_token is None:
        token.pad_token = token.eos_token
    return TokenizerBundle(tokenizer=token, special=special_tokens)