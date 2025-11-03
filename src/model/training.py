from __future__ import annotations

import os, random
import numpy as np
import torch
import mlflow
import hydra

from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from pathlib import Path
from transformers import AutModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback

from model.tokenizer import load_tokenizer
from model.sft_dataset import SFTJsonlDataset

def set_deterministic(seed: int, determinsitc: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if determinsitc:
        torch.use_deterministic_alogrithms(True, warn_only=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
@hydra.main(config_path='../../configs', config_name='model', version_base='1.3')
def main(cfg: DictConfig):
    print('Config:\n', OmegaConf.to_yaml(cfg))
    
    set_deterministic(cfg.seed, cfg.deterministic)
    mlflow.set_experiment('sft')
    with mlflow.start_run():
        tok_bundle = load_tokenizer(cfg.backbone_name, cfg.special_tokens)
        tokenizer = tok_bundle.tokenizer
        
        model = AutModelForCausalLM.from_pretrained(cfg.backbone_name)
        model.resize_token_embeddings(len(tokenizer))
        
        data_cfg = OmegaConf.load(Path(hydra.utils.get_original_cwd()) / "configs" / "data.yaml")
        train_ds = SFTJsonlDataset(Path(data_cfg.sft_data.train_file), tokenizer, cfg.special_tokens, cfg.sft.max_seq_len)
        val_ds = SFTJsonlDataset(Path(data_cfg.sft_data.val_file), tokenizer, cfg.special_tokens, cfg.sft.max_seq_len)
        
        args = TrainingArguments(
            output_dir=cfg.sft.output_dir,
            per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
            per_device_eval_batch_size=max(1, cfg.sft.per_device_train_batch_size),
            gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
            learning_rate=cfg.sft.learning_rate,
            weight_decay=cfg.sft.weight_decay,
            num_train_epochs=cfg.sft.num_train_epochs,
            warmup_ratio=cfg.sft.warmup_ratio,
            fp16=cfg.sft.fp16,
            bf16=cfg.sft.bf16,
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
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.sft.early_stopping_patience)],
        )

        trainer.train()
        trainer.save_model(cfg.sft.output_dir)
        tokenizer.save_pretrained(cfg.sft.output_dir)
        
if __name__ == '__main__':
    main()
