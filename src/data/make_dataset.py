from __future__ import annotations

import argparse, json, csv, gzip, random, sys
from pathlib import Path
from typing import Iterable, Dict, Any, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .transforms import to_sft_record, to_ppo_record, PlanPrefs

class LifeCoachData(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.rows.append(json.loads(line))
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        ex = self.rows[idx]
        input = f"PROFILE: {ex['user_profile']} GOAL: {ex['goal_description']}"
        target = _flatten_target(ex['plan'], ex['nudges'])
        return {'input_text': input, 'target_text': target}

class PPOJSONLDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer: PreTrainedTokenizerBase, max_query_len: int = 1024, max_response_len: int = 512):
        self.path = Path(jsonl_path)
        self.rows = list(_read_any(self.path))
        self.tokenizer = tokenizer
        self.max_q = max_query_len
        self.max_r = max_response_len
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.rows)

    def _tok(self, text: str, max_len: int) -> torch.LongTensor:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
            add_special_tokens=False,  # already embedded role tokens in the string
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        q = r["query"]
        q_ids = self._tok(q, self.max_q)

        item: Dict[str, Any] = {
            "query": q,
            "query_tensors": q_ids,
        }

        # Optional teacher/reference response (for KL ref or warm-start)
        if "reference_response" in r and r["reference_response"]:
            item["reference_response"] = r["reference_response"]
            item["reference_response_tensors"] = self._tok(r["reference_response"], self.max_r)

        # Optional gold response (if present)
        if "response" in r and r["response"]:
            item["response"] = r["response"]
            item["response_tensors"] = self._tok(r["response"], self.max_r)

        return item
    
def _flatten_target(plan: List[Dict[str, Any]], nudges: List[str]) -> str:
    plan_str = " | ".join([f"{s['step_id']}. {s['description']} [{s['energy']}/{s['location']}/{s['minutes']}m]" for s in plan])
    nudges_str = " | ".join(nudges)
    return f"PLAN: {plan_str} || NUDGES: {nudges_str}"

def _read_any(path: Path) -> Iterable[Dict[str, Any]]:
    open_fn = gzip.open if path.suffix == ".gz" else open
    if path.suffix in {".jsonl", ".jsonl.gz"}:
        with open_fn(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix in {".json", ".gz"}:
        with open_fn(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for r in data:
                    yield r
            else:
                yield data
    elif path.suffix in {".csv"}:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def ppo_collate(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def _pad(seqs: List[torch.LongTensor]) -> torch.LongTensor:
        if not seqs:
            return torch.empty(0, dtype=torch.long)
        max_len = max(s.size(0) for s in seqs)
        out = torch.full(len(seqs), max_len, fill_value=tokenizer.pad_token_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out
    
    queries = [b["query"] for b in batch]
    q_tensors = [b["query_tensors"] for b in batch]
    batched = {"queries": queries, "query_tensors": _pad(q_tensors)}

    # Optional fields
    if "response_tensors" in batch[0]:
        batched["responses"] = [b.get("response", "") for b in batch]
        batched["response_tensors"] = _pad([b["response_tensors"] for b in batch])

    if "reference_response_tensors" in batch[0]:
        batched["reference_responses"] = [b.get("reference_response", "") for b in batch]
        batched["reference_response_tensors"] = _pad([b["reference_response_tensors"] for b in batch])

    return batched

def main():
    ap = argparse.ArgumentParser(description="Make datasets for SFT and PPO (no OASST).")
    ap.add_argument("--raw", required=True, nargs="+", help="Path(s) to raw JSON/JSONL/CSV[.gz].")
    ap.add_argument("--out-sft", required=True, help="Output JSONL for SFT.")
    ap.add_argument("--out-ppo", required=True, help="Output JSONL for PPO.")
    ap.add_argument("--timezone", default="UTC")
    ap.add_argument("--focus-block", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    prefs = PlanPrefs(timezone=args.timezone, focus_blocks_min=args.focus_block)

    raw_items: List[Dict[str, Any]] = []
    for p in args.raw:
        raw_items.extend(list(_read_any(Path(p))))

    # light clean
    def _norm(i: Dict[str, Any]) -> Dict[str, Any]:
        i = dict(i)
        i["type"] = (i.get("type") or "task").lower()
        if isinstance(i.get("steps"), str):
            i["steps"] = [s.strip() for s in i["steps"].split("|") if s.strip()]
        return i

    raw_items = [_norm(i) for i in raw_items]

    sft = [to_sft_record(i, prefs) for i in raw_items]
    ppo = [to_ppo_record(i, prefs) for i in raw_items]

    _write_jsonl(Path(args.out_sft), sft)
    _write_jsonl(Path(args.out_ppo), ppo)

    print(f"SFT -> {args.out_sft}  |  PPO -> {args.out_ppo}  |  N={len(raw_items)}")

if __name__ == "__main__":
    main()
