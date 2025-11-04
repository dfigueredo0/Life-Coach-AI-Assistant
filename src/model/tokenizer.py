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
    tok = AutoTokenizer.from_pretrained(backbone_name, use_fast=True, padding_side="right", truncation_side="right")
    tok.add_special_tokens(SPECIAL_TOKENS)
    if tok.eos_token is None:
        tok.eos_token = "<|endoftext|>"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    special = {
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "system": "<|system|>",
        "end": "<|endoftext|>",
    }
    
    return TokenizerBundle(tokenizer=tok, special=special)