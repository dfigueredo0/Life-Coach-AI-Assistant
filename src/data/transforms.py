from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Iterable, Tuple
import datetime as dt

SPECIAL = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end": "<|endoftext|>",
}

@dataclass
class PlanPrefs:
    timezone: str = "UTC"
    work_start_h: int = 9
    work_end_h: int = 18
    focus_blocks_min: int = 50
    
def _coerce_dt(x: str | None) -> str | None:
    if not x: 
        return None
    try:
        return dt.datetime.fromisoformat(x).isoformat()
    except Exception:
        return x
    
def itemize_user_prompt(item: Dict[str, Any]) -> str:
    t = item.get("type", "task")
    if t == "task":
        bits = [
            f"Goal: {item.get('title', '')}",
            f"Details: {item.get('description', '')}",
            f"Steps: {', '.join(item.get('steps', [])[:10])}",
            f"Energy: {item.get('energy', 'unknown')}",
            f"Location: {item.get('location', 'any')}",
            f"Due: {_coerce_dt(item.get('due') or item.get('deadline'))}"
        ]
        return "\n".join([b for b in bits if b and b.endswith(': ')])
    if t == "event":
        return f"I have an event: {item.get('title','(untitled)')} from {_coerce_dt(item.get('start'))} to {_coerce_dt(item.get('end'))}. Help me plan around it."
    if t == "note":
        return f"Note: {item.get('text','')}\nTurn this into actionable steps."
    return f"Item: {item}"

def assistant_plan_reply(item: Dict[str, Any], prefs: PlanPrefs) -> str:
    # Deterministic, rule-based template. Replace with model later if desired.
    steps = item.get("steps") or []
    if not steps and item.get("description"):
        # naive step split by punctuation
        steps = [s.strip() for s in item["description"].replace(";", ".").split(".") if s.strip()][:5]
    steps = steps[:8] if steps else ["Define scope", "Draft", "Review", "Finalize"]

    energy = item.get("energy", "medium")
    block_min = max(25, min(120, int(item.get("duration_min") or 50)))
    tags = f"[energy:{energy}|location:{item.get('location','any')}]"
    plan_lines = [f"- ({block_min}m) {s}" for s in steps]

    return "\n".join([
        "Here’s a concrete plan with time blocks and tags.",
        f"Preferences: TZ={prefs.timezone}, focus_block={prefs.focus_blocks_min}m",
        tags,
        *plan_lines,
        "I’ll schedule around existing events and confirm before writing to your calendar.",
    ])

def to_conversation_turns(item: Dict[str, Any], prefs: PlanPrefs) -> List[Dict[str,str]]:
    sys = (
        "You are a life coach assistant. Decompose vague goals into small steps, "
        "tag steps by energy/focus/location, avoid guilt, respect focus mode, and confirm before scheduling."
    )
    user = itemize_user_prompt(item)
    assistant = assistant_plan_reply(item, prefs)
    return [
        {"role": "system", "text": sys},
        {"role": "user", "text": user},
        {"role": "assistant", "text": assistant},
    ]

def render_chat(turns: List[Dict[str,str]]) -> str:
    out = []
    for t in turns:
        out.append(SPECIAL[t["role"]] + t["text"] + SPECIAL["end"])
    return "".join(out)

def to_sft_record(item: Dict[str, Any], prefs: PlanPrefs) -> Dict[str, Any]:
    turns = to_conversation_turns(item, prefs)
    return {"turns": turns, "text": render_chat(turns)}

def to_ppo_record(item: Dict[str, Any], prefs: PlanPrefs) -> Dict[str, Any]:
    turns = to_conversation_turns(item, prefs)
    prompt = SPECIAL["system"] + turns[0]["text"] + SPECIAL["end"] + SPECIAL["user"] + turns[1]["text"] + SPECIAL["end"]
    return {"query": prompt, "reference_response": turns[2]["text"]}