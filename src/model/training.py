from __future__ import annotations

import os, argparse, random, importlib, contextlib
import numpy as np
import torch
import mlflow
import hydra

from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback

from data.collators import DataCollator
from model.tokenizer import load_tokenizer
from model.sft_dataset import SFTJsonlDataset

def set_deterministic(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def _has_torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _has_directml() -> bool:
    return importlib.util.find_spec("torch_directml") is not None 
        
def setup_accelertor(cfg):
    """
    Returns:
      device: torch.device or torch_directml device
      autocast_ctx: context manager for AMP autocast (no-op if backend missing)
      scaler: torch.amp.GradScaler or dummy (None) when AMP off
      notes: string describing backend
    """
    use_amp_flag = bool(getattr(cfg, "trainer", None) and getattr(cfg.trainer, "amp", False))

    if _has_torch_cuda():  # NVIDIA CUDA or AMD ROCm (HIP) — both surface as torch.cuda
        import torch
        device = torch.device("cuda")
        # Backend perf toggles (safe on ROCm; TF32 will be ignored on AMD)
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        use_amp = bool(cfg.trainer.amp and torch.cuda.is_available())
        scaler = torch.amp.GradScaler(enabled=use_amp)
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)
        notes = "torch.cuda (CUDA/ROCm)"
        
        return device, autocast_ctx, scaler, notes
    elif _has_directml():  # Windows fallback for AMD via DirectML
        import torch_directml
        device = torch_directml.device()
        scaler = None  # AMP not supported via torch-directml
        autocast_ctx =  contextlib.nullcontext()
        notes = "torch-directml (DirectML)"
        return device, autocast_ctx, scaler, notes
    else:
        import torch
        device = torch.device("cpu")
        scaler = None
        autocast_ctx = contextlib.nullcontext()
        notes = "CPU"
        return device, autocast_ctx, scaler, notes

def make_fast_loaders(ds_train, ds_val, train_bs, eval_bs, pin=True):
    """
    Builds high-throughput DataLoaders.
    """
    from torch.utils.data import DataLoader
    common = dict(
        num_workers=8,                 # tune 6–12 if you have spare cores
        pin_memory=pin,                # True iff GPU
        persistent_workers=True,       # keeps workers alive
        prefetch_factor=4,             # 2–6 depending on sample cost
        drop_last=False,
    )
    dl_train = DataLoader(ds_train, batch_size=train_bs, shuffle=True, **common)
    dl_val   = DataLoader(ds_val,   batch_size=eval_bs,  shuffle=False, **common)
    return dl_train, dl_val

def to_device(x, device):
    # Use non_blocking only when the destination is torch.cuda device
    import torch
    non_blocking = isinstance(device, torch.device) and device.type == "cuda"
    if hasattr(x, "to"):
        return x.to(device, non_blocking=non_blocking)
    return x

@hydra.main(config_path='../../configs', config_name='model', version_base='1.3')
def main(cfg: DictConfig):    
    print('Config:\n', OmegaConf.to_yaml(cfg))
    set_deterministic(cfg.seed, cfg.deterministic)
    
    device, autocast_ctx, scaler, notes = setup_accelertor(cfg)
    is_dml = ("directml" in notes.lower())
    is_cuda = ("cuda" in notes.lower())
    print(f"[backend] {notes}")
    
    if is_dml:
        # ensure mixed precision is off with DML
        if getattr(cfg.sft, "fp16", False) or getattr(cfg.sft, "bf16", False):
            print("[warning] Disabling fp16/bf16 on DirectML")
            cfg.sft.fp16 = False
            cfg.sft.bf16 = False
    
    mlflow.set_experiment('sft')
    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        tok_bundle = load_tokenizer(cfg.backbone_name, cfg.special_tokens)
        tokenizer = tok_bundle.tokenizer
        specials = tok_bundle.special
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(cfg.backbone_name)
        model.resize_token_embeddings(len(tokenizer))
        
        data_cfg = OmegaConf.load(Path(hydra.utils.get_original_cwd()) / "configs" / "data.yaml")
        train_ds = SFTJsonlDataset(Path(data_cfg.sft_data.train_file), tokenizer, specials, cfg.sft.max_seq_len)
        val_ds = SFTJsonlDataset(Path(data_cfg.sft_data.val_file), tokenizer, specials, cfg.sft.max_seq_len)
        
        collator = DataCollator(tokenizer=tokenizer, max_length=cfg.sft.max_seq_len, mask_user=True)

        out_dir = Path(cfg.sft.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if not is_dml:
            args = TrainingArguments(
                output_dir=str(out_dir),
                per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
                per_device_eval_batch_size=max(1, cfg.sft.per_device_train_batch_size),
                gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
                learning_rate=cfg.sft.learning_rate,
                weight_decay=cfg.sft.weight_decay,
                num_train_epochs=cfg.sft.num_train_epochs,
                warmup_ratio=cfg.sft.warmup_ratio,
                fp16=cfg.sft.fp16 if is_cuda else False,  # only allow on CUDA
                bf16=cfg.sft.bf16 if is_cuda else False,
                gradient_checkpointing=cfg.sft.gradient_checkpointing,
                logging_steps=cfg.sft.logging_steps,
                evaluation_strategy="steps",
                eval_steps=cfg.sft.eval_steps,
                save_steps=cfg.sft.save_steps,
                save_total_limit=2,
                load_best_model_at_end=True,
                report_to=["mlflow"],
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=collator,
                tokenizer=tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.sft.early_stopping_patience)],
                    )

            mlflow.set_experiment('sft')
            with mlflow.start_run():
                mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
                trainer.train(resume_from_checkpoint=True)
                trainer.save_model(str(out_dir))
                tokenizer.save_pretrained(str(out_dir))
                mlflow.log_artifacts(str(out_dir))
            return
        
    print("[info] Using custom training loop for DirectML")
    from torch.utils.data import DataLoader

    per_device_bs = cfg.sft.per_device_train_batch_size
    eval_bs = max(1, per_device_bs)

    train_dl = DataLoader(train_ds, batch_size=per_device_bs, shuffle=True, num_workers=4,
                          collate_fn=collator, pin_memory=False, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=eval_bs,  shuffle=False, num_workers=4,
                          collate_fn=collator, pin_memory=False, persistent_workers=True)

    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.sft.learning_rate, weight_decay=cfg.sft.weight_decay)

    def _run_eval():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                tot += float(loss.detach().cpu()); n += 1
        model.train()
        return tot / max(1, n)

    total_steps = (len(train_dl) * cfg.sft.num_train_epochs)
    log_every   = max(1, cfg.sft.logging_steps)
    eval_every  = max(1, cfg.sft.eval_steps)

    best_val = float("inf")
    out_dir = Path(cfg.sft.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment('sft')
    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        step = 0
        for epoch in range(cfg.sft.num_train_epochs):
            for batch in train_dl:
                # move to DML device
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast_ctx:
                    outputs = model(**batch)
                    loss = outputs.loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.sft.max_grad_norm if "max_grad_norm" in cfg.sft else 1.0)
                opt.step()

                step += 1
                if step % log_every == 0:
                    print(f"epoch {epoch} step {step}/{total_steps}  loss {loss.item():.4f}")

                if step % eval_every == 0:
                    val_loss = _run_eval()
                    print(f"[val] step {step}  loss {val_loss:.4f}")
                    mlflow.log_metric("train_loss", float(loss.item()), step=step)
                    mlflow.log_metric("val_loss", float(val_loss), step=step)
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(model.state_dict(), out_dir / "best_dml.pt")

        # final save (HF format requires config too)
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        mlflow.log_artifacts(str(out_dir))
        
if __name__ == '__main__':
    main()
