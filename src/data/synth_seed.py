from __future__ import annotations
import json, random
from pathlib import Path

ENERGY = ["low","med","high"]
LOCATIONS = ["home","office","outdoors"]
GOALS = [
    "Write a 2-page report on {topic}",
    "Clean the {room}",
    "Study {subject} for the quiz",
    "Prepare a healthy meal plan for the week",
]
TOPICS = ["budgeting","DA algorithms","physics notes","email backlog"]
ROOMS = ["kitchen","bathroom","bedroom"]
SUBJECTS = ["math","history","biology"]

def make_raw(n: int, seed: int = 42):
    random.seed(seed)
    rows = []
    for _ in range(n):
        goal = random.choice(GOALS).format(
            topic=random.choice(TOPICS),
            room=random.choice(ROOMS),
            subject=random.choice(SUBJECTS)
        )
        steps = [f"{i+1}. {name} part {i+1}" for i, name in enumerate(["Outline","Do","Review","Polish","Submit"][:random.randint(3,5)])]
        rows.append({
            # minimal raw fields your transforms can handle
            "type": "task",
            "goal": goal,
            "profile": {
                "energy_baseline": random.choice(ENERGY),
                "sensory_load": random.choice(["low","med","high"]),
                "schedule_volatility": random.choice(["low","med","high"])
            },
            "steps": steps,
            "nudge_templates": [
                "Start now: {step}", "Halfway check: {step}",
                "Wrap up in 5 min: {step}", "Missed? Resume with: {step}"
            ]
        })
    return rows

def main():
    out = Path("data/raw_synth.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = make_raw(10000, seed=42)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out}  N={len(rows)}")

if __name__ == "__main__":
    main()
