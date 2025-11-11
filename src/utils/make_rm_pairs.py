from __future__ import annotations
import json, argparse, random
from pathlib import Path

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="artifacts/sft/eval/samples.jsonl")  # <- correct default
    ap.add_argument("--out", default="data/rm_pairs.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = list(read_jsonl(Path(args.samples)))

    src = Path(args.samples)
    if not src.exists():
        raise FileNotFoundError(f"Samples not found: {src}. Run lc-eval first.")

    # Build simple pairwise prefs: (chosen = prediction, rejected = corrupted prediction)
    out = []
    for r in rows:
        prompt = r.get("prompt", "")
        chosen = r.get("prediction", "")
        rejected = chosen

        # Corrupt by truncation or shuffling tokens
        toks = chosen.split()
        if len(toks) > 6 and rng.random() < 0.5:
            rejected = " ".join(toks[: len(toks)//2])
        else:
            rng.shuffle(toks)
            rejected = " ".join(toks)

        out.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}  N={len(out)}")

if __name__ == "__main__":
    main()
