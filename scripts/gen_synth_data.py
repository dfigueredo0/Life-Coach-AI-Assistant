import json, random, os
from typing import List, Dict

from src.data.schema import Example, Step

ENERGY = ["low","med","high"]
LOCATIONS = ["home","office","outdoors"]
TOPICS = ["budgeting","DA algorithms","physics notes","email backlog","bathroom","kitchen","CV update"]
NUDGE_TEMPLATES = [
    "Start now: {step}",
    "Halfway check: {step}",
    "Wrap up in 5 min: {step}",
    "Missed? Resume with: {step}"
]

GOALS = [
    "Write a 2-page report on {topic}",
    "Clean the {room}",
    "Study {subject} for the quiz",
    "Prepare a healthy meal plan for the week",
]

def make_example() -> Dict:
    volatility = random.choices(ENERGY)
    profile = {
        "energy_baseline": random.choice(ENERGY),
        "sensory_load": random.choice(ENERGY),
        "schedule_volatility": volatility
    }
    goal = random.choice(GOALS).format(topic=random.choice(TOPICS), room=random.choice(LOCATIONS), subject=random.choice(['math', 'history', 'biology']))
    n_steps = random.randint(3, 6)
    steps = []
    for i in range(n_steps):
        steps.append(Step(
            step_id=i+1,
            description=f"{['Outline','Do','Review','Polish','Submit'][i%5]} part {i+1}",
            energy=random.choice(ENERGY),
            location=random.choice(LOCATIONS),
            minutes=random.choice([10,15,20,25,30])
        ).__dict__)
    nudges = [t.format(steps=s['description']) for t in NUDGE_TEMPLATES]
    return {
        "user_profile": profile, 
        "goal_description": goal, 
        "plan": steps, 
        "nudges": nudges
    }
    
def generate(out_dir: str, n_total: int = 1000, seed: int = 42):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    data = [make_example() for _ in range(n_total)]
    random.shuffle(data)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    splits = {
        "train.jsonl": data[:n_train],
        "val.jsonl": data[n_train:n_train + n_val],
        "test.jsonl": data[:n_train+n_val]
    }
    for name, items in splits.items():
        with open(os.path.join(out_dir, name), 'w', encoding='utf-8') as f:
            for ex in items:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    generate(out_dir="data", n_total=10000)    