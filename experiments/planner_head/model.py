"""Qwen3.6-27B text tower, a Kev-style LoRA, and a pointer head.

Qwen3.6-27B is a hybrid of full attention and Gated DeltaNet, so a question cannot be
packed beside others under an attention mask. Each training row is one causal sequence:
the state, then the single next-tool question. The head reads the hidden state at the
question's decide token and at each option's closing token.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

KEV_ROOT = Path("/cfsdata/agent/kev")
if str(KEV_ROOT) not in sys.path:
    sys.path.insert(0, str(KEV_ROOT))

from kev.model import encode, is_hybrid, load_tokenizer, rows_of  # noqa: E402

from pointer import PointerHead  # noqa: E402

# Kev's hybrid target list (kev/model.py, lora_targets="all").
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
]


class ToolDecisionModel(nn.Module):
    def __init__(self, base: str, device: str, lora_rank: int = 16, head_dim: int = 256):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3_5ForConditionalGeneration

        self.tokenizer = load_tokenizer(base)
        self.owner = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16)
        # The traces are text. Drop the vision tower and the token head so they are not trained.
        self.owner.model.visual = None
        self.owner.lm_head = None
        text = self.owner.model.language_model
        if not is_hybrid(text.config):
            raise ValueError(f"{base} has no linear-attention layers; this trainer expects the Qwen3.6 hybrid tower")
        text.config.use_cache = False
        lora = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
        )
        self.lm = get_peft_model(text, lora)
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    def encode_row(self, record: dict) -> dict:
        from kev.data import materialize

        internal = materialize(record)
        enc = encode(self.tokenizer, internal)
        state_ids, state_pos, branches = rows_of(enc)
        if len(branches) != 1:
            raise ValueError("each row must contain exactly one question")
        branch = branches[0]
        return {
            "ids": state_ids + branch["ids"],
            "pos": state_pos + branch["pos"],
            "decide": len(state_ids) + branch["decide"],
            "opts": [len(state_ids) + offset for offset in branch["opts"]],
            "label": internal["questions"][0]["label"],
        }

    def logits(self, encoded: dict) -> torch.Tensor:
        ids = torch.tensor([encoded["ids"]], device=self.device)
        pos = torch.tensor([encoded["pos"]], device=self.device)
        hidden = self.lm(input_ids=ids, position_ids=pos, attention_mask=torch.ones_like(ids)).last_hidden_state[0].float()
        return self.head(hidden[encoded["decide"]], hidden[encoded["opts"]])

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
