import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class RMDataset(Dataset):
    file: Path
    tokenizer: AutoTokenizer
    max_len: int

    def __post_init__(self):
        self.items = []
        with open(self.file, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __getitem__(self, idx):
        ex = self.items[idx]
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        return {
            "chosen": self.tokenizer(f"{prompt}\n{chosen}", truncation=True, max_length=self.max_len, return_tensors="pt"),
            "rejected": self.tokenizer(f"{prompt}\n{rejected}", truncation=True, max_length=self.max_len, return_tensors="pt")
        }

    def __len__(self):
        return len(self.items)
    
def main(cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.backbone_name,
            num_labels=1
        ).to(device)

        ds = RMDataset(Path("data/rm/rm_train.jsonl"), tokenizer, cfg.rm.max_seq_len)
        dl = DataLoader(ds, batch_size=cfg.rm.per_device_train_batch_size, shuffle=True)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.rm.learning_rate)
        steps = len(dl) * cfg.rm.num_train_epochs

        sched = get_scheduler("linear", opt, steps, int(steps * cfg.rm.warmup_ratio))

        for epoch in range(cfg.rm.num_train_epochs):
            for batch in tqdm(dl):
                chosen = {k: v.squeeze(1).to(device) for k, v in batch["chosen"].items()}
                rejected = {k: v.squeeze(1).to(device) for k, v in batch["rejected"].items()}

                chosen_scores = model(**chosen).logits
                rejected_scores = model(**rejected).logits

                loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()

                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad()

        Path(cfg.rm.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(cfg.rm.output_dir)
        tokenizer.save_pretrained(cfg.rm.output_dir)