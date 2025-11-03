from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class RMComponents:
    tokenizer: Any
    model: Any
    
def load_reward_model(backbone: str, num_labels: int = 1) -> RMComponents:
    tok = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=num_labels)
    return RMComponents(tokenizer=tok, model=model)

def compute_rewards(model, tokenizer, texts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        toks = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        out = model(**toks)
        logits = out.logits.squeeze(-1)
        return logits