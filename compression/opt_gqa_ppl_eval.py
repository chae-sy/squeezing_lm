# opt_gqa_test.py
import math
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    OPTForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import os
from transformers import AutoConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class GQAAttention(nn.Module):
    """
    Drop-in replacement for OPT self-attn that uses:
      - H query heads (same as original),
      - G key/value heads (G divides H),
    and repeats K/V groups to match Q heads for attention computation.
    Conversion copies Q and OUT projections; K/V are recreated with out_features = G * head_dim.

    NOTE: For compatibility, we return/persist group-level KV in past_key_values,
    then repeat at compute time. This keeps memory light in our custom greedy generator.
    """
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "H must be divisible by G"
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        assert self.head_dim * self.num_q_heads == embed_dim
        self.num_heads = num_q_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_dropout_p = dropout

    @staticmethod
    def _shape(x, B, T, n_heads, head_dim):
        return x.view(B, T, n_heads, head_dim).permute(0, 2, 1, 0 + 2)  # (B, nH, T, d)

    def forward(
    self,
    hidden_states: torch.Tensor,                     # (B, T, D)
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,   # usually None for OPT eval
    layer_head_mask: Optional[torch.Tensor] = None,  # ignored
    output_attentions: bool = False,
    use_cache: bool = False,
    key_value_states: Optional[torch.Tensor] = None, # self-attn only
    position_ids: Optional[torch.Tensor] = None,     # ignored
    **kwargs,                                        # future-proof
):
        if key_value_states is not None:
            raise NotImplementedError("GQAAttention supports self-attention only.")

        import math
        B, T, D = hidden_states.shape
        H = self.num_q_heads
        G = self.num_kv_heads
        d = self.head_dim

        # ---- 1) Projections ----
        q = self.q_proj(hidden_states)   # (B, T, D)
        k = self.k_proj(hidden_states)   # (B, T, G*d)
        v = self.v_proj(hidden_states)   # (B, T, G*d)

        q = q.view(B, T, H, d).permute(0, 2, 1, 3)  # (B, H, Tq, d)
        k = k.view(B, T, G, d).permute(0, 2, 1, 3)  # (B, G, Tk_new, d)
        v = v.view(B, T, G, d).permute(0, 2, 1, 3)  # (B, G, Tk_new, d)

        # ---- 2) Append past (accept None, (), (k,v)) ----
        k_past = v_past = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                k_past, v_past = past_key_value
        if (k_past is not None) and (v_past is not None):
            # k_past/v_past: (B, G, Tk_past, d)
            k = torch.cat([k_past, k], dim=2)  # (B, G, Tk_total, d)
            v = torch.cat([v_past, v], dim=2)

        present_key_value = (k, v) if use_cache else None  # store group-level cache

        # ---- 3) Repeat KV groups to match Q heads ----
        k_r = k.repeat_interleave(self.group_size, dim=1)  # (B, H, Tk, d)
        v_r = v.repeat_interleave(self.group_size, dim=1)  # (B, H, Tk, d)

        # ---- 4) Build additive causal + (optional) padding mask ----
        B_, H_, Tq, d_ = q.shape
        Tk = k_r.size(2)
        past_len = Tk - Tq  # = 0 when no cache

        # causal_add: (Tq, Tk) with 0 for keep, -inf for disallow (future)
        i = torch.arange(Tq, device=q.device)[:, None] + past_len
        j = torch.arange(Tk, device=q.device)[None, :]
        causal_add = (j > i).to(q.dtype) * (-1e9)                        # (Tq, Tk)

        # Optional 2-D padding mask (rare in this eval path): 1=keep, 0=pad
        pad_add = None
        if (attention_mask is not None) and (attention_mask.dim() == 2):
            pad_add = (1.0 - attention_mask[:, None, :].to(q.dtype)) * (-1e9)  # (B, 1, Tk)

        # ---- 5) Scores + masks → softmax → dropout → weighted sum ----
        scores = torch.matmul(q, k_r.transpose(-2, -1)) / math.sqrt(d)   # (B, H, Tq, Tk)
        scores = scores + causal_add.view(1, 1, Tq, Tk)                  # broadcast over B,H
        if pad_add is not None:
            scores = scores + pad_add.unsqueeze(1).expand(B, H, Tq, Tk)  # broadcast over H,Tq

        attn_probs = torch.softmax(scores, dim=-1)
        if self.training and getattr(self, "attn_dropout_p", 0.0) > 0:
            attn_probs = F.dropout(attn_probs, p=self.attn_dropout_p)

        attn_output = torch.matmul(attn_probs, v_r)                       # (B, H, Tq, d)

        # ---- 6) Merge heads & output proj ----
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, Tq, D)  # (B, T, D)
        attn_output = self.out_proj(attn_output)

        # HF expects: 2-tuple when not caching; 3-tuple when caching
        if use_cache:
            return attn_output, None, present_key_value
        else:
            return attn_output, None


        

    @classmethod
    def from_opt_attention(cls, opt_attn_module, num_kv_heads: int):
        """Build GQAAttention from an OPT self_attn module; perform mean-pool conversion for K/V."""
        H = opt_attn_module.num_heads
        D = opt_attn_module.embed_dim
        d = D // H
        G = num_kv_heads
        assert H % G == 0, f"H={H} must be divisible by G={G}"

        gqa = cls(embed_dim=D, num_q_heads=H, num_kv_heads=G, dropout=getattr(opt_attn_module, "dropout", 0.0))

        # Copy Q and OUT as-is
        gqa.q_proj.weight.data.copy_(opt_attn_module.q_proj.weight.data)
        if opt_attn_module.q_proj.bias is not None:
            gqa.q_proj.bias.data.copy_(opt_attn_module.q_proj.bias.data)

        gqa.out_proj.weight.data.copy_(opt_attn_module.out_proj.weight.data)
        if opt_attn_module.out_proj.bias is not None:
            gqa.out_proj.bias.data.copy_(opt_attn_module.out_proj.bias.data)

        # Mean-pool K/V projection matrices within groups (paper method)
        def pool_kv(linear: nn.Linear) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            W = linear.weight.data  # (D_out, D_in) = (D, D)
            b = linear.bias.data if linear.bias is not None else None
            W = W.view(H, d, D)  # per-head rows
            if b is not None:
                b = b.view(H, d)

            group_size = H // G
            Ws = []
            bs = []
            for g in range(G):
                hs = slice(g * group_size, (g + 1) * group_size)
                Wg = W[hs].mean(dim=0)        # (d, D)
                Ws.append(Wg)
                if b is not None:
                    bg = b[hs].mean(dim=0)    # (d,)
                    bs.append(bg)
            W_new = torch.cat(Ws, dim=0)      # (G*d, D)
            b_new = torch.cat(bs, dim=0) if b is not None else None
            return W_new, b_new

        Wk, bk = pool_kv(opt_attn_module.k_proj)
        Wv, bv = pool_kv(opt_attn_module.v_proj)
        gqa.k_proj.weight.data.copy_(Wk)
        gqa.v_proj.weight.data.copy_(Wv)
        if bk is not None: gqa.k_proj.bias.data.copy_(bk)
        if bv is not None: gqa.v_proj.bias.data.copy_(bv)
        return gqa


def patch_opt_to_gqa(model: OPTForCausalLM, groups: int):
    for layer in model.model.decoder.layers:
        old = layer.self_attn
        gqa = GQAAttention.from_opt_attention(old, num_kv_heads=groups)
        layer.self_attn = gqa
    # Mark on config for reference
    if not hasattr(model.config, "num_key_value_heads"):
        model.config.num_key_value_heads = groups
    else:
        model.config.num_key_value_heads = groups
    return model



def get_dataset(name: str):
    if name.lower() in ["wikitext-2", "wikitext2", "wt2"]:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        text_column = "text"
    elif name.lower() in ["wikitext-103", "wikitext103", "wt103"]:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        text_column = "text"
    else:
        # small, easy fallback
        ds = load_dataset("tiny_shakespeare")
        text_column = "text"
    return ds, text_column


def tokenize_dataset(ds, tokenizer, text_column, block_size=1024):
    def _tok(batch):
        return tokenizer(batch[text_column], return_attention_mask=False)
    tokenized = ds.map(_tok, batched=True, remove_columns=[text_column])
    # Group into block_size chunks for causal LM
    def _group(examples):
        concat = []
        for lst in examples["input_ids"]:
            concat.extend(lst)
        total_len = (len(concat) // block_size) * block_size
        result = {
            "input_ids": [concat[i:i+block_size] for i in range(0, total_len, block_size)]
        }
        return result
    out = tokenized.map(_group, batched=True)
    out.set_format(type="torch", columns=["input_ids"])
    return out



@torch.no_grad()
def eval_ppl(model, tokenizer, dataset, batch_size=8, max_batches=None, device="cuda"):
    import math, torch
    model.eval().to(device)

    loader = torch.utils.data.DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        # (B, T)
        input_ids = batch["input_ids"].to(device)

        # Make labels that match HF's causal shift exactly:
        # shift_logits = logits[..., :-1, :]
        # shift_labels = labels[..., 1:]
        # So we set last label to -100 to exclude it from loss.
        labels = input_ids.clone()
        labels[:, -1] = -100

        out = model(input_ids=input_ids, labels=labels)

        # out.loss is mean over tokens != -100 in THIS batch
        loss = out.loss
        num_toks = (labels != -100).sum().item()

        # Safety checks to catch attention/mask issues early
        if not torch.isfinite(loss):
            print(f"[eval_ppl] Non-finite loss detected at batch {i}: {loss.item()}")
            return float("inf")

        total_loss += loss.item() * num_toks
        total_tokens += num_toks

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_loss / total_tokens
    ppl = math.exp(avg_nll)
    return ppl



@torch.no_grad()
def speed_test(model, tokenizer, gen_input_len=512, gen_new_tokens=256, device="cuda"):
    model.eval().to(device)
    # Build a synthetic prompt
    prompt = "Once upon a time, " * (gen_input_len // 5)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[:, :gen_input_len]

    # Custom greedy loop that reuses the model's returned past_key_values
    # (with group-level caches provided by our GQAAttention)
    max_new = gen_new_tokens
    cur_ids = input_ids
    past = None

    # Warmup
    _ = model(input_ids=cur_ids, use_cache=True)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for step in range(max_new):
        if past is None:
            out = model(input_ids=cur_ids, use_cache=True)
        else:
            out = model(input_ids=cur_ids[:, -1:], use_cache=True, past_key_values=past)

        logits = out.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
        cur_ids = torch.cat([cur_ids, next_id], dim=1)
        past = out.past_key_values

    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    toks_per_s = gen_new_tokens / dt
    return toks_per_s, tokenizer.decode(cur_ids[0, -gen_new_tokens:])



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--load_dir", default=None, help="Load a previously saved (converted/uptrained) model")
    ap.add_argument("--save_dir", default=None, help="Where to save converted or trained model")

    ap.add_argument("--groups", type=int, default=6, help="Number of GQA groups (G). Use 1 for MQA.")
    ap.add_argument("--convert_only", action="store_true")

    ap.add_argument("--train", action="store_true")
    ap.add_argument("--max_steps", type=int, default=2000, help="Uptraining steps (paper used alpha≈0.05)")
    ap.add_argument("--dataset", default="wikitext-2")

    ap.add_argument("--eval_ppl", action="store_true")
    ap.add_argument("--eval_speed", action="store_true")
    ap.add_argument("--gen_input_len", type=int, default=512)
    ap.add_argument("--gen_new_tokens", type=int, default=256)
    args = ap.parse_args()

    # Load
    tokenizer = AutoTokenizer.from_pretrained(args.load_dir or args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_dir:
        # 1) tokenizer from the saved dir (keeps same vocab/eos)
        tokenizer = AutoTokenizer.from_pretrained(args.load_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2) figure out GQA groups (prefer the saved config; fallback to CLI)
        try:
            saved_cfg = AutoConfig.from_pretrained(args.load_dir)
            groups = getattr(saved_cfg, "num_key_value_heads", None) or args.groups
        except Exception:
            groups = args.groups

        # 3) build base OPT from the original model id, then patch to GQA
        base_id = args.model  # e.g., "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
        model = patch_opt_to_gqa(model, groups=groups)

        # 4) load state dict from your saved GQA checkpoint
        state = None
        bin_path = os.path.join(args.load_dir, "pytorch_model.bin")
        sft_path = os.path.join(args.load_dir, "model.safetensors")
        sft_idx = os.path.join(args.load_dir, "model.safetensors.index.json")

        if os.path.isfile(bin_path):
            state = torch.load(bin_path, map_location="cpu")
        else:
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise RuntimeError(
                    "No pytorch_model.bin found and safetensors not installed. "
                    "pip install safetensors or save with .bin."
                )
            if os.path.isfile(sft_path):
                state = load_file(sft_path)
            elif os.path.isfile(sft_idx):
                # merge sharded safetensors
                import json
                with open(sft_idx, "r", encoding="utf-8") as f:
                    idx = json.load(f)
                state = {}
                for shard_path in idx["weight_map"].values():
                    shard_full = os.path.join(args.load_dir, shard_path)
                    state.update(load_file(shard_full))
            else:
                raise RuntimeError(f"No weights found in {args.load_dir}")

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded GQA checkpoint with groups={groups}. "
            f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        # original cold load path
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)


     # Convert to GQA only if we're not loading a pre-converted checkpoint.
    if not args.load_dir:
        print(f"Converting model to GQA with G={args.groups} groups...")
        model = patch_opt_to_gqa(model, groups=args.groups)

    # Save conversion if requested
    if args.convert_only and args.save_dir:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Saved converted GQA model to: {args.save_dir}")
        return

    # Train (uptraining)
    if args.train:
        ds, text_col = get_dataset(args.dataset)
        tokenized = tokenize_dataset(ds, tokenizer, text_col, block_size=1024)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=args.save_dir or "./opt_gqa_run",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            weight_decay=0.0,
            warmup_steps=0,
            max_steps=args.max_steps,
            eval_strategy="steps",     # <- evaluate
            eval_steps=200,                  # every 200 steps
            save_strategy="steps",
            save_steps=1000,                 # checkpoint every 1k steps
            logging_steps=50,
            save_total_limit=2,
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
            fp16=not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],   # <-- REQUIRED when evaluating
            data_collator=collator,
        )

        trainer.train()
        if args.save_dir:
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            print(f"Saved uptrained GQA model to: {args.save_dir}")

    # Evaluate PPL
    if args.eval_ppl:
        ds, text_col = get_dataset(args.dataset)
        tokenized = tokenize_dataset(ds, tokenizer, text_col, block_size=1024)
        ppl = eval_ppl(
    model, tokenizer, tokenized,
    batch_size=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_batches=50
)
        print(f"Perplexity (approx): {ppl:.3f}")

    # Speed test
    if args.eval_speed:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tps, sample = speed_test(
            model, tokenizer,
            gen_input_len=args.gen_input_len,
            gen_new_tokens=args.gen_new_tokens,
            device=device
        )
        print(f"Greedy decode speed: {tps:.2f} toks/s  (input {args.gen_input_len}, new {args.gen_new_tokens})")
        print("Sample tail:", sample[:120].replace("\n", " ") + "...")
        

if __name__ == "__main__":
    main()
