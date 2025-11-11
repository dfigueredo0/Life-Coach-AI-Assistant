from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

@dataclass
class SFTJsonlDataset(Dataset):
    file: Path
    tokenizer: PreTrainedTokenizerBase
    special: Dict[str, str]
    max_len: int

    def __init__(self, file: Path, tokenizer: PreTrainedTokenizerBase, special: Dict[str, str], max_len: int):
        self.items: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.special = special or {}
        self.max_len = max_len
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

    # --- Helpers ---
    def _tok(self, key: str, default: str = "") -> str:
        return self.special.get(key, default)

    def _format_schema_text(self, ex: Dict[str, Any]) -> str:
        """Already linearized supervision."""
        t = ex.get("text")
        return t if isinstance(t, str) else str(ex)

    def _format_schema_a(self, ex: Dict[str, Any]) -> str:
        """Your older format: goal/profile/steps."""
        user = self._tok("user", "<|user|>")
        assistant = self._tok("assistant", "<|assistant|>")
        end = self._tok("end", "<|endoftext|>")

        goal = ex.get("goal", "")
        profile = ex.get("profile", "")
        steps = ex.get("steps", "")

        prompt = f"{user} Goal: {goal}\nProfile: {profile}\nSteps: {steps}"
        resp = (
            f"{assistant} I understand your goal. "
            f"Here is a structured plan with nudges included.\n"
            f"{steps}\n{end}"
        )
        return f"{prompt}\n{resp}"

    def _format_schema_b(self, ex: Dict[str, Any]) -> str:
        """Synthetic pipeline format: goal_description/user_profile/plan/nudges."""
        user = self._tok("user", "<|user|>")
        assistant = self._tok("assistant", "<|assistant|>")
        end = self._tok("end", "<|endoftext|>")

        goal = ex.get("goal_description", "")
        profile = ex.get("user_profile", "")
        plan = ex.get("plan", [])
        nudges = ex.get("nudges", [])

        # compact plan line
        if isinstance(plan, list):
            plan_str = " | ".join(
                f"{s.get('step_id', i+1)}. {s.get('description','')}"
                f" [{s.get('energy','')}/{s.get('location','')}/{s.get('minutes','')}m]"
                for i, s in enumerate(plan)
            )
        else:
            plan_str = str(plan)

        nudge_str = " | ".join(map(str, nudges)) if isinstance(nudges, list) else str(nudges)

        prompt = f"{user} Goal: {goal}\nProfile: {profile}"
        resp = f"{assistant} PLAN: {plan_str} || NUDGES: {nudge_str}\n{end}"
        return f"{prompt}\n{resp}"

    def _format_turns(self, ex: Dict[str, Any]) -> str:
        """Fallback: linearize 'turns' with special tokens."""
        sys_tok = self._tok("system", "<|system|>")
        usr_tok = self._tok("user", "<|user|>")
        as_tok  = self._tok("assistant", "<|assistant|>")
        end_tok = self._tok("end", "<|endoftext|>")
        parts = []
        for t in ex.get("turns", []):
            role = t.get("role", "")
            content = t.get("content", "")
            if role == "system":    parts.append(f"{sys_tok}{content}{end_tok}")
            elif role == "user":    parts.append(f"{usr_tok}{content}{end_tok}")
            elif role == "assistant": parts.append(f"{as_tok}{content}{end_tok}")
        return "".join(parts) if parts else str(ex)

    # --- Dataset protocol ---
    def __getitem__(self, idx: int) -> Dict[str, str]:
        ex = self.items[idx]

        if "text" in ex and isinstance(ex["text"], str):
            text = self._format_schema_text(ex)
        elif "goal" in ex or "profile" in ex or "steps" in ex:
            text = self._format_schema_a(ex)
        elif "goal_description" in ex or "user_profile" in ex or "plan" in ex:
            text = self._format_schema_b(ex)
        elif "turns" in ex:
            text = self._format_turns(ex)
        else:
            text = str(ex)

        # ALWAYS return dict with "text" so the collator can tokenize it
        return {"text": text}

    def __len__(self) -> int:
        return len(self.items)
