from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from .reward_model import compute_rewards, load_reward_model

@dataclass
class PPOArtifacts:
    ppo_trainer: PPOTrainer
    rm_name: str
    
def build(policy_name: str, rm_name: str, cfg: Dict) -> PPOArtifacts:
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(policy_name)
    ppo_config = PPOConfig(
        learning_rate=cfg['learning_rate'], 
        batch_size=cfg['batch_size'], 
        mini_batch_size=cfg['min_batch_size'], 
        ppo_epochs=cfg['ppo_epochs'],
        target_kl=cfg['target_kl']
    )
    rm = load_reward_model(rm_name)
    trainer = PPOTrainer(ppo_config, policy, None, tokenizer=None)
    return PPOArtifacts(ppo_trainer=trainer, rm_name=rm_name)

def ppo_step(art: PPOArtifacts, queries: List[str], responses: List[str]):
    texts = [q + r for q, r in zip(queries, responses)]
    rewards = compute_rewards(load_reward_model(art.rm_name).model, load_reward_model(art.rm_name).tokenizer, texts)
    
    rewards = rewards.detach()
    # Need to pass model inpuits (queries/responses tokenized) to PPOTrainer.step
    # Here only show interface; dataset adaptor would supply query_tensors/response_tensors.