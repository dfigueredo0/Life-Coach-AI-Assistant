# src/scripts/eval_sft.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf

def rouge_l(hyp: str, ref: str) -> float:
    a, b = hyp.split(), ref.split()
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = a[i-1]
        row = dp[i]
        prev = dp[i-1]
        for j in range(1, m+1):
            row[j] = prev[j-1] + 1 if ai == b[j-1] else (row[j-1] if row[j-1]>prev[j] else prev[j])
    lcs = dp[n][m]
    return (2*lcs) / max(1, (n + m))

def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def pick_text(r: Dict) -> str:
    if "text" in r and isinstance(r["text"], str): return r["text"]
    if "turns" in r:
        # very light linearization
        return "".join(t.get("content","") for t in r["turns"])
    return str(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/sft")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--sample", action="store_true")   # otherwise greedy
    args = ap.parse_args()

    data_cfg = OmegaConf.load(args.data)
    test_path = Path(data_cfg.sft_data.test_file)
    test_items = load_jsonl(test_path)

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt)
    model.eval(); model.to("cpu")  # generation on CPU is fine for small models

    preds, refs, rows = [], [], []
    for r in test_items[:200]:  # cap for speed; bump later
        prompt = pick_text(r)
        ref = pick_text(r)      # if your SFT has gold targets, use them; else compare to input format as proxy
        inp = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.sample,
                temperature=0.9 if args.sample else None,
                top_p=0.9 if args.sample else None,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        preds.append(text); refs.append(ref)
        rows.append({"prompt": prompt, "prediction": text, "reference": ref})

    # metrics
    rouge = sum(rouge_l(p, r) for p, r in zip(preds, refs)) / max(1, len(preds))
    format_rate = sum(("PLAN:" in p and "NUDGES:" in p) for p in preds) / max(1, len(preds))

    out_dir = Path(args.ckpt) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"rougeL": rouge, "format_rate": format_rate}, f, ensure_ascii=False, indent=2)

    print(f"ROUGE-L: {rouge:.3f} | format_rate: {format_rate:.2%} | wrote {out_dir}")

if __name__ == "__main__":
    main()
