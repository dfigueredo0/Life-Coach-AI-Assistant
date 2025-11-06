import json, random
from pathlib import Path

def main():
    src = Path("data/sft.jsonl")
    rows = [json.loads(l) for l in src.open("r", encoding="utf-8") if l.strip()]
    random.seed(42); random.shuffle(rows)
    n = len(rows); n_train = int(0.70*n); n_val = int(0.15*n)

    splits = {
    "data/sft_train.jsonl": rows[:n_train],
    "data/sft_val.jsonl":   rows[n_train:n_train+n_val],
    "data/sft_test.jsonl":  rows[n_train+n_val:],
    }

    for p, r in splits.items():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for x in r: f.write(json.dumps(x, ensure_ascii=False)+"\n")

    print({k: len(v) for k, v in splits.items()})
