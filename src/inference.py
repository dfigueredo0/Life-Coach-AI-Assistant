from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from transformers import AutoModelForCasualML
from model.tokenizer import load_tokenizer

PROMPT_TEMPLATES = {
    "plan": "{sys}{user}Goal: {goal}\nConstraints: {constraints}\nPlease produce a concise, step-by-step plan.{end}{assistant}",
    "nudge": "{sys}{user}Task: {task}\nEnergy: {energy}\nGive a gentle, context-aware nudge.{end}{assistant}",
}

@dataclass
class InferenceBundle:
    model: Any
    tokenizer: Any
    special: Dict[str, str]
    
def load_inference(backbone: str, special: Dict[str, str]) -> InferenceBundle:
    tb = load_tokenizer(backbone, special)
    model = AutoModelForCasualML.from_pretrained(backbone)
    model.resize_token_embeddings(len(tb.tokenizer))
    return InferenceBundle(model=model, tokenizer=tb.tokenizer, special=special)

def build_prompt(template_key: str, special: Dict[str, str], **kwargs) -> str:
    t = PROMPT_TEMPLATES[template_key]
    return t.format(
        sys=f"{special['system']}You are a helpful life coach.{special['end']}",
        user=special["user"],
        assistant=special["assistant"],
        end=special["end"],
        **kwargs,
    )