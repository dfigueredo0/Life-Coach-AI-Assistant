from __future__ import annotations
import argparse, torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from data.rm_dataset import RMPairs, collate_rm

class RewardHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.scorer = nn.Linear(hidden, 1)

    def forward(self, last_hidden_state, attention_mask):
        cls = last_hidden_state[:, 0, :]  # [B, H]
        return self.scorer(cls).squeeze(-1)

class RewardModel(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        self.head = RewardHead(self.encoder.config.hidden_size)

    def score(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state, attention_mask)

    def forward(self, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask):
        pos = self.score(pos_input_ids, pos_attention_mask)
        neg = self.score(neg_input_ids, neg_attention_mask)
        return pos, neg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/rm_pairs.jsonl")
    ap.add_argument("--base", default="distilroberta-base")
    ap.add_argument("--out", default="artifacts/rm")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # DML is unreliable here; use CPU/CUDA
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    ds = RMPairs(args.pairs, tok, max_len=512)
    dl = DataLoader(ds, batch_size=args.bsz, shuffle=True, collate_fn=collate_rm, num_workers=0)

    model = RewardModel(args.base).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MarginRankingLoss(margin=0.1)

    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            pos, neg = model(**batch)
            target = torch.ones_like(pos)
            loss = loss_fn(pos, neg, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.4f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Save encoder + head (separately)
    tok.save_pretrained(out)
    model.encoder.save_pretrained(out)
    torch.save(model.head.state_dict(), out / "head.pt")
    print(f"Saved RM to {out}")

if __name__ == "__main__":
    main()
