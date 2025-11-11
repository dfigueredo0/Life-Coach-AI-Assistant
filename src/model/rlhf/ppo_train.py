from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

class PromptDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("prompt") or row.get("query")
                if isinstance(text, str) and text.strip():
                    self.rows.append(text)
                    
    def __len__(self): 
        return len(self.rows)
    
    def __getitem__(self, i): 
        return self.rows[i]

class RewardModel(torch.nn.Module):
    def __init__(self, rm_dir: Path, device):
        super().__init__()
        self.device = device

        self.encoder = AutoModel.from_pretrained(rm_dir)
        hidden = self.encoder.config.hidden_size

        self.head = torch.nn.Linear(hidden, 1)
        sd = torch.load(rm_dir / "head.pt", map_location="cpu", weights_only=True)

        if "scorer.weight" in sd:
            sd = {"weight": sd["scorer.weight"], "bias": sd["scorer.bias"]}

        self.head.load_state_dict(sd, strict=True)

        self.encoder.to(self.device)
        self.head.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(rm_dir)

    @torch.no_grad()
    def score_pairs(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        texts = [p + " " + r for p, r in zip(prompts, responses)]
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.encoder(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)

def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
             import torch_directml
             return torch_directml.device()
        except Exception:
            return torch.device("cpu")
    # fallback
    return torch.device("cpu")

def to_id_tensors(tokenizer, texts: List[str], device: torch.device) -> List[torch.LongTensor]:
    out: List[torch.LongTensor] = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=True).input_ids[0]
        out.append(ids.to(device))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="Path to SFT checkpoint (artifacts/sft)")
    ap.add_argument("--rm", required=True, help="Path to reward model (artifacts/rm)")
    ap.add_argument("--prompts", required=True, help="JSONL with {'prompt' or 'query': ...}")
    ap.add_argument("--out", required=True, help="Output PPO directory")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mini_batch", type=int, default=1)
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[PPO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.sft, use_fast=True, padding_side="left", truncation_side="left")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft)
    policy.pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.pretrained_model.config.pad_token_id = tokenizer.pad_token_id
    policy.to(device)
    ref_model.to(device if device.type == "cuda" else "cpu")

    ds = PromptDataset(args.prompts)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    config = PPOConfig(
        batch_size=args.batch,
        mini_batch_size=args.mini_batch,
        ppo_epochs=2,
        learning_rate=2e-6,
        target_kl=0.2,
        init_kl_coef=0.05,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        whiten_rewards=False,
        optimize_cuda_cache=False,
        seed=42,
    )

    trainer = PPOTrainer(
        config=config,
        model=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=ds
    )
    trainer.tokenizer.padding_side = "left"

    rm = RewardModel(Path(args.rm), device)
    rm = RewardModel(Path(args.rm), device)
    rm.tokenizer.padding_side = "left"
    if rm.tokenizer.pad_token is None:
        rm.tokenizer.pad_token = rm.tokenizer.eos_token
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)

    gen_kwargs = dict(
        max_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n=== PPO Epoch {epoch+1}/{args.epochs} ===")
        for batch_prompts in dl:
            prompts_text = list(batch_prompts)
            
            enc = tokenizer(
                prompts_text,
                padding=True,            # left padding because padding_side='left'
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            
            enc = {k: v.to(device) for k, v in enc.items()}
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]

            gen = policy.pretrained_model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                **gen_kwargs,
            )

            prompt_lens = attn.sum(dim=1)              
            responses = []
            queries = []
            for i in range(gen.size(0)):
                Lp = prompt_lens[i].item()
                queries.append(input_ids[i, -Lp:].detach())
                responses.append(gen[i, input_ids.size(1):].detach())

            responses_text = tokenizer.batch_decode(responses, skip_special_tokens=True)

            with torch.no_grad():
                raw_r = rm.score_pairs(prompts_text, responses_text).to(device)

            r_std = raw_r.std().clamp_min(1e-3)
            reward_tensor = torch.tanh((raw_r - raw_r.mean()) / r_std) * 0.1
            rewards = [r.unsqueeze(0) for r in reward_tensor]

            trainer.step(queries, responses, rewards)

        save_dir = out_dir / f"epoch_{epoch+1}"
        policy.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"[PPO] Saved: {save_dir}")

    print("[PPO] Training complete.")

if __name__ == "__main__":
    main()