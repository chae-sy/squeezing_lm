# opt_gqa_lambada_eval.py
import os
import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    OPTForCausalLM,
)
from datasets import load_dataset


class GQAAttention(nn.Module):
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "H must be divisible by G"
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        assert self.head_dim * self.num_q_heads == embed_dim
        self.num_heads = num_q_heads
        self.dropout = dropout

        self.q_proj  = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj  = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj  = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout_p = dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        key_value_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if key_value_states is not None:
            raise NotImplementedError("GQAAttention supports self-attention only.")

        import math
        B, T, D = hidden_states.shape
        H = self.num_q_heads
        G = self.num_kv_heads
        d = self.head_dim

        q = self.q_proj(hidden_states)  # (B, T, D)
        k = self.k_proj(hidden_states)  # (B, T, G*d)
        v = self.v_proj(hidden_states)  # (B, T, G*d)

        q = q.view(B, T, H, d).permute(0, 2, 1, 3)  # (B, H, T, d)
        k = k.view(B, T, G, d).permute(0, 2, 1, 3)  # (B, G, T, d)
        v = v.view(B, T, G, d).permute(0, 2, 1, 3)  # (B, G, T, d)

        k_past = v_past = None
        if past_key_value is not None and isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
            k_past, v_past = past_key_value
        if (k_past is not None) and (v_past is not None):
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        k_r = k.repeat_interleave(self.group_size, dim=1)  # (B, H, Tk, d)
        v_r = v.repeat_interleave(self.group_size, dim=1)  # (B, H, Tk, d)

        B_, H_, Tq, _ = q.shape
        Tk = k_r.size(2)
        past_len = Tk - Tq

        i = torch.arange(Tq, device=q.device)[:, None] + past_len
        j = torch.arange(Tk, device=q.device)[None, :]
        causal_add = (j > i).to(q.dtype) * (-1e9)

        pad_add = None
        if (attention_mask is not None) and (attention_mask.dim() == 2):
            pad_add = (1.0 - attention_mask[:, None, :].to(q.dtype)) * (-1e9)  # (B, 1, Tk)

        scores = torch.matmul(q, k_r.transpose(-2, -1)) / math.sqrt(d)  # (B, H, Tq, Tk)
        scores = scores + causal_add.view(1, 1, Tq, Tk)
        if pad_add is not None:
            scores = scores + pad_add.unsqueeze(1).expand(B, H, Tq, Tk)

        attn_probs = torch.softmax(scores, dim=-1)
        if self.training and getattr(self, "attn_dropout_p", 0.0) > 0:
            attn_probs = F.dropout(attn_probs, p=self.attn_dropout_p)

        attn_output = torch.matmul(attn_probs, v_r)  # (B, H, Tq, d)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, Tq, D)
        attn_output = self.out_proj(attn_output)

        if use_cache:
            return attn_output, None, present_key_value
        else:
            return attn_output, None

    @classmethod
    def from_opt_attention(cls, opt_attn_module, num_kv_heads: int):
        H = opt_attn_module.num_heads
        D = opt_attn_module.embed_dim
        d = D // H
        G = num_kv_heads
        assert H % G == 0, f"H={H} must be divisible by G={G}"

        gqa = cls(embed_dim=D, num_q_heads=H, num_kv_heads=G, dropout=getattr(opt_attn_module, "dropout", 0.0))

        # copy Q / OUT
        gqa.q_proj.weight.data.copy_(opt_attn_module.q_proj.weight.data)
        if opt_attn_module.q_proj.bias is not None:
            gqa.q_proj.bias.data.copy_(opt_attn_module.q_proj.bias.data)
        gqa.out_proj.weight.data.copy_(opt_attn_module.out_proj.weight.data)
        if opt_attn_module.out_proj.bias is not None:
            gqa.out_proj.bias.data.copy_(opt_attn_module.out_proj.bias.data)

        # mean-pool within groups for K/V
        def pool_kv(linear: nn.Linear):
            W = linear.weight.data  # (D, D)
            b = linear.bias.data if linear.bias is not None else None
            W = W.view(H, d, D)
            if b is not None:
                b = b.view(H, d)

            group_size = H // G
            Ws, bs = [], []
            for g in range(G):
                hs = slice(g * group_size, (g + 1) * group_size)
                Wg = W[hs].mean(dim=0)           # (d, D)
                Ws.append(Wg)
                if b is not None:
                    bs.append(b[hs].mean(dim=0)) # (d,)
            W_new = torch.cat(Ws, dim=0)         # (G*d, D)
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
    if not hasattr(model.config, "num_key_value_heads"):
        model.config.num_key_value_heads = groups
    else:
        model.config.num_key_value_heads = groups
    return model


class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total if total > 0 else 0.0
        return acc


def load_gqa_model(load_dir: str, base_id: str = "facebook/opt-125m", groups_fallback: int = 6):
    # tokenizer from the saved dir (keeps same vocab/eos)
    tokenizer = AutoTokenizer.from_pretrained(load_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # infer groups from saved config, fallback to CLI default
    try:
        saved_cfg = AutoConfig.from_pretrained(load_dir)
        groups = getattr(saved_cfg, "num_key_value_heads", None) or groups_fallback
    except Exception:
        groups = groups_fallback

    # build base OPT, then patch to GQA
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32)
    model = patch_opt_to_gqa(model, groups=groups)

    # load weights from dir (supports .bin or safetensors)
    state = None
    bin_path = os.path.join(load_dir, "pytorch_model.bin")
    sft_path = os.path.join(load_dir, "model.safetensors")
    sft_idx  = os.path.join(load_dir, "model.safetensors.index.json")

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
            import json
            with open(sft_idx, "r", encoding="utf-8") as f:
                idx = json.load(f)
            state = {}
            for shard_path in idx["weight_map"].values():
                shard_full = os.path.join(load_dir, shard_path)
                state.update(load_file(shard_full))
        else:
            raise RuntimeError(f"No weights found in {load_dir}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] groups={groups}  missing={len(missing)}  unexpected={len(unexpected)}")
    return model, tokenizer, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_dir", required=True, help="Directory with uptrained/converted GQA OPT-125M")
    ap.add_argument("--model", default="facebook/opt-125m", help="Base model id to instantiate before patching")
    ap.add_argument("--groups", type=int, default=6, help="Fallback groups if not in saved config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_samples", type=int, default=1000, help="Evaluate on first N examples of LAMBADA")
    ap.add_argument("--cuda_visible_devices", default=None, help="Optional: set CUDA_VISIBLE_DEVICES env var (e.g., '2')")
    args = ap.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # 1) Load model/tokenizer
    model, tokenizer, groups = load_gqa_model(args.load_dir, base_id=args.model, groups_fallback=args.groups)
    model.to(args.device)

    # 2) Load LAMBADA and slice
    split = "validation" if args.max_samples is None else f"validation[:{args.max_samples}]"
    dataset = load_dataset("lambada", split=split)

    # 3) Evaluate last-word accuracy (your Evaluator)
    evaluator = Evaluator(dataset, tokenizer, device=args.device)
    acc = evaluator.evaluate(model)
    print(f"LAMBADA last-word accuracy (G={groups}, N={len(dataset)}): {acc*100:.2f}%")

if __name__ == "__main__":
    main()
