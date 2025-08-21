# opt_combined.py
# Pipeline:
#   load ./opt125m-gqa3-up  -> ensure GQA=3
#   -> SmoothQuant calibrate  -> PTQ W4A4
#   -> LoRA adapters  -> QAT (adapters only, W4A4 fake-quant)
#
# pip install --upgrade torch transformers datasets accelerate safetensors

import math
import argparse
from typing import Optional, Tuple, Dict
from inspect import signature

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # expose only GPU:2 to this process


# =============== Misc utils ===============

def get_device(name: Optional[str] = None) -> str:
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_wikitext2():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    return ds, "text"


def tokenize_dataset(ds, text_column, tokenizer, block_size=1024):
    def _tok(batch):
        return tokenizer(batch[text_column], return_attention_mask=False)
    tokenized = ds.map(_tok, batched=True, remove_columns=[text_column])

    def _group(examples):
        concat = []
        for ids in examples["input_ids"]:
            concat.extend(ids)
        total = (len(concat) // block_size) * block_size
        blocks = [concat[i:i+block_size] for i in range(0, total, block_size)]
        return {"input_ids": blocks}
    out = tokenized.map(_group, batched=True)
    out.set_format(type="torch", columns=["input_ids"])
    return out


@torch.no_grad()
def eval_ppl(model, tokenizer, tokenized, device="cuda", batch_size=8, max_batches=50):
    model.eval().to(device)
    loader = torch.utils.data.DataLoader(tokenized["validation"], batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        labels = x.clone()
        labels[:, -1] = -100  # match causal shift
        out = model(input_ids=x, labels=labels)
        loss = out.loss
        if not torch.isfinite(loss):
            print(f"[eval] non-finite loss at batch {i}: {loss.item()}")
            return float("inf")
        toks = (labels != -100).sum().item()
        total_loss += loss.item() * toks
        total_tokens += toks
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


# =============== GQA attention patch ===============

class GQAAttention(nn.Module):
    """
    OPT self-attn with grouped KV heads (G). Q has H heads, KV have G (< H).
    """
    def __init__(self, old_attn, num_kv_heads: int):
        super().__init__()
        self.embed_dim   = old_attn.embed_dim
        self.num_q_heads = old_attn.num_heads
        self.head_dim    = old_attn.head_dim
        self.dropout     = old_attn.dropout

        self.num_kv_heads = num_kv_heads
        assert self.num_q_heads % self.num_kv_heads == 0, "H must be divisible by G"
        self.group_size = self.num_q_heads // self.num_kv_heads

        self.q_proj = old_attn.q_proj
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.out_proj = old_attn.out_proj
        self.num_heads = self.num_q_heads

    @staticmethod
    def from_opt_attention(old_attn, num_kv_heads: int):
        gqa = GQAAttention(old_attn, num_kv_heads=num_kv_heads)
        with torch.no_grad():
            H = old_attn.num_heads
            d = old_attn.head_dim
            D = old_attn.embed_dim
            G = num_kv_heads
            group = H // G

            def _pool(W: torch.Tensor, b: Optional[torch.Tensor]):
                Wv = W.view(H, d, D).view(G, group, d, D).mean(dim=1).view(G * d, D).contiguous()
                if b is None: return Wv, None
                bv = b.view(H, d).view(G, group, d).mean(dim=1).reshape(G * d)
                return Wv, bv

            Wk, bk = _pool(old_attn.k_proj.weight.detach().clone(),
                           old_attn.k_proj.bias.detach().clone() if old_attn.k_proj.bias is not None else None)
            Wv, bv = _pool(old_attn.v_proj.weight.detach().clone(),
                           old_attn.v_proj.bias.detach().clone() if old_attn.v_proj.bias is not None else None)

            gqa.k_proj.weight.copy_(Wk)
            if gqa.k_proj.bias is not None and bk is not None: gqa.k_proj.bias.copy_(bk)
            gqa.v_proj.weight.copy_(Wv)
            if gqa.v_proj.bias is not None and bv is not None: gqa.v_proj.bias.copy_(bv)
        return gqa

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,   # usually None with labels
        layer_head_mask: Optional[torch.Tensor] = None,  # ignored
        output_attentions: bool = False,
        use_cache: bool = False,
        key_value_states: Optional[torch.Tensor] = None, # self-attn only
        position_ids: Optional[torch.Tensor] = None,     # ignored
        **kwargs,
    ):
        if key_value_states is not None:
            raise NotImplementedError("GQAAttention supports self-attention only.")

        B, T, D = hidden_states.shape
        H = self.num_q_heads
        G = self.num_kv_heads
        d = self.head_dim

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, T, H, d).permute(0, 2, 1, 3)  # (B,H,T,d)
        k = k.view(B, T, G, d).permute(0, 2, 1, 3)  # (B,G,T,d)
        v = v.view(B, T, G, d).permute(0, 2, 1, 3)  # (B,G,T,d)

        k_past = v_past = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                k_past, v_past = past_key_value
        if (k_past is not None) and (v_past is not None):
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        present_kv = (k, v) if use_cache else None

        k_r = k.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,d)
        v_r = v.repeat_interleave(self.group_size, dim=1)

        Tq = q.size(2); Tk = k_r.size(2)
        past_len = Tk - Tq
        i = torch.arange(Tq, device=q.device)[:, None] + past_len
        j = torch.arange(Tk, device=q.device)[None, :]
        causal_add = (j > i).to(q.dtype) * (-1e9)

        scores = torch.matmul(q, k_r.transpose(-2, -1)) / math.sqrt(d)
        scores = scores + causal_add.view(1, 1, Tq, Tk)

        attn = torch.softmax(scores, dim=-1)
        if self.training and self.dropout and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        y = torch.matmul(attn, v_r)

        y = y.permute(0, 2, 1, 3).contiguous().view(B, Tq, D)
        y = self.out_proj(y)

        if use_cache:
            return y, None, present_kv
        return y, None


def is_model_gqa(model) -> bool:
    try:
        n_kv = getattr(model.config, "num_key_value_heads", None)
        n_h  = getattr(model.config, "num_attention_heads", None)
        return (n_kv is not None) and (n_kv != n_h)
    except Exception:
        return False


def patch_opt_to_gqa(model, groups: int):
    for layer in model.model.decoder.layers:
        old = layer.self_attn
        if isinstance(old, GQAAttention):
            continue
        layer.self_attn = GQAAttention.from_opt_attention(old, num_kv_heads=groups)
    model.config.num_key_value_heads = groups
    return model


# =============== Quantization (SmoothQuant + QuantLinear) ===============

def qmin_qmax(bits: int):
    qmax = (1 << (bits - 1)) - 1
    qmin = - (1 << (bits - 1))
    return qmin, qmax

def sym_quantize(x: torch.Tensor, bits: int, per_channel_dim: Optional[int] = None):
    assert bits in (4, 6, 8)
    qmin, qmax = qmin_qmax(bits)
    if per_channel_dim is None:
        s = x.abs().amax().clamp(min=1e-8) / qmax
        x_q = (x / s).round().clamp(qmin, qmax).to(torch.int8)
        return x_q, s
    else:
        x_perm = x.transpose(0, per_channel_dim).contiguous()
        shp = x_perm.shape
        flat = x_perm.view(shp[0], -1)
        s = flat.abs().amax(dim=1).clamp(min=1e-8) / qmax
        x_q = (x_perm / s[:, None]).round().clamp(qmin, qmax).to(torch.int8)
        x_q = x_q.view(shp).transpose(0, per_channel_dim).contiguous()
        return x_q, s

def sym_dequantize(x_q: torch.Tensor, scale: torch.Tensor, per_channel_dim: Optional[int] = None):
    if per_channel_dim is None:
        return x_q.float() * scale
    view_shape = [1] * x_q.dim()
    view_shape[per_channel_dim] = -1
    return x_q.float() * scale.view(view_shape)

class QuantLinear(nn.Module):
    def __init__(self, base: nn.Linear, w_bits=4, a_bits=4, act_pre_scale: Optional[torch.Tensor]=None):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.register_buffer("act_pre_scale", None if act_pre_scale is None else act_pre_scale.detach().clone())
        W = base.weight.detach()
        W_q, W_s = sym_quantize(W, bits=w_bits, per_channel_dim=0)
        self.register_buffer("W_q", W_q)
        self.register_buffer("W_s", W_s)

    def forward(self, x):
        if self.act_pre_scale is not None:
            x = x / self.act_pre_scale
        if self.a_bits < 16:
            x_q, x_s = sym_quantize(x, bits=self.a_bits, per_channel_dim=None)
            x = sym_dequantize(x_q, x_s)
        W = sym_dequantize(self.W_q, self.W_s, per_channel_dim=0)
        return F.linear(x, W, self.bias)

def replace_linear_with_quant(mod: nn.Module, w_bits=4, a_bits=4, smooth_scales: Optional[Dict[str, torch.Tensor]]=None, prefix=""):
    for name, child in list(mod.named_children()):
        full = f"{prefix}{name}" if prefix == "" else f"{prefix}.{name}"
        if isinstance(child, nn.Linear):
            aps = smooth_scales.get(full) if smooth_scales else None
            setattr(mod, name, QuantLinear(child, w_bits=w_bits, a_bits=a_bits, act_pre_scale=aps))
        else:
            replace_linear_with_quant(child, w_bits=w_bits, a_bits=a_bits, smooth_scales=smooth_scales, prefix=full)

@torch.no_grad()
def collect_act_max_per_input_channel(model: nn.Module, tokenizer, dataset, device, num_batches=64, block_size=1024):
    model.eval().to(device)
    stats: Dict[str, torch.Tensor] = {}
    handles = []

    def _hook(name):
        def fn(m, inp, out):
            x = inp[0]
            x = x.reshape(-1, x.shape[-1]).abs()
            cur = x.max(dim=0).values
            if name not in stats: stats[name] = cur.detach().cpu()
            else: stats[name] = torch.maximum(stats[name], cur.detach().cpu())
        return fn

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(_hook(name)))

    ds, text_col = dataset
    tok = tokenize_dataset(ds, text_col, tokenizer, block_size=block_size)
    loader = torch.utils.data.DataLoader(tok["train"], batch_size=8, shuffle=False)
    for i, b in enumerate(loader):
        if i >= num_batches: break
        ids = b["input_ids"].to(device)
        _ = model(input_ids=ids)

    for h in handles: h.remove()
    return stats

def compute_smooth_scales(model: nn.Module, act_max: Dict[str, torch.Tensor], alpha=0.5, eps=1e-6):
    scales = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name in act_max:
            W = m.weight.detach().cpu()
            W_col_max = W.abs().amax(dim=0) + eps
            A_max = act_max[name].to(W_col_max.device) + eps
            s = (A_max / W_col_max).pow(alpha)
            act_pre_scale = (1.0 / s).clamp(min=1e-6)
            scales[name] = act_pre_scale
    return scales
# =============== LoRA (QAT) ===============

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, lora_alpha=16, lora_dropout=0.0, a_bits=4, w_bits=4):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False

        self.r = r
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        self.a_bits = a_bits
        self.w_bits = w_bits

    def forward(self, x):
        y = self.base(x)
        if self.a_bits < 16:
            x_q, x_s = sym_quantize(x, bits=self.a_bits, per_channel_dim=None)
            x = sym_dequantize(x_q, x_s)
        x = self.dropout(x)

        A_w = self.A.weight; B_w = self.B.weight
        if self.w_bits < 16:
            A_q, A_s = sym_quantize(A_w, bits=self.w_bits, per_channel_dim=0)
            B_q, B_s = sym_quantize(B_w, bits=self.w_bits, per_channel_dim=0)
            A_w = sym_dequantize(A_q, A_s, per_channel_dim=0)
            B_w = sym_dequantize(B_q, B_s, per_channel_dim=0)

        delta = F.linear(F.linear(x, A_w), B_w) * self.scaling
        return y + delta


# ---------- LoRA injection that supports QuantLinear ----------
def inject_lora(model: nn.Module, r=8, lora_alpha=16, lora_dropout=0.0,
                a_bits=4, w_bits=4,
                targets=("q_proj","k_proj","v_proj","out_proj","fc1","fc2")):
    """
    Wrap both nn.Linear and QuantLinear with LoRALinear on selected module names.
    """
    target_types = (nn.Linear, QuantLinear)

    def _recurse(mod, prefix=""):
        for name, child in list(mod.named_children()):
            full = f"{prefix}{name}" if prefix == "" else f"{prefix}.{name}"
            if isinstance(child, target_types) and any(t in full for t in targets):
                setattr(
                    mod, name,
                    LoRALinear(child, r=r, lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout, a_bits=a_bits, w_bits=w_bits)
                )
            else:
                _recurse(child, full)

    _recurse(model)
    return model


# ---------- Progressive activation-bits warmup (A8 -> A4) ----------
from transformers import TrainerCallback

class ActivationBitsWarmup(TrainerCallback):
    """
    Start QAT with A=8 for stability, then switch to A=4 at `switch_step`.
    Also tightens grad clipping mid-run.
    """
    def __init__(self, switch_step=1500, final_a_bits=4):
        self.switch_step = switch_step
        self.final_a_bits = final_a_bits
        self.switched = False

    def _set_a_bits(self, model, bits):
        # set a_bits in QuantLinear and LoRALinear
        for m in model.modules():
            if hasattr(m, "a_bits"):
                m.a_bits = bits

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.switched and state.global_step >= self.switch_step:
            self._set_a_bits(kwargs["model"], self.final_a_bits)
            # tighten grad clip after the switch (optional but helpful)
            if hasattr(args, "max_grad_norm") and args.max_grad_norm:
                args.max_grad_norm = min(args.max_grad_norm, 0.5)
            self.switched = True
            print(f"[ABitsWarmup] Switched activations to A{self.final_a_bits} at step {state.global_step}.")
        return control


# ---------- Stable LoRA QAT trainer ----------
from inspect import signature

def train_lora_qat(model, tokenizer, ds, device, steps=6000, lr=2e-5,
                   batch=8, grad_accum=8, block_size=1024, save_dir=None):
    model.to(device)
    tokenized = tokenize_dataset(*ds, tokenizer, block_size=block_size)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ta_kwargs = dict(
        output_dir=save_dir or "./opt_combined_qat",
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.10,            # longer warmup
        max_steps=steps,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=steps,
        save_total_limit=1,
        max_grad_norm=0.8,            # tighter clipping
        optim="adafactor",            # very stable for adapters
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        fp16=not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
    )
    # version guards
    if "lr_scheduler_type" in signature(TrainingArguments).parameters:
        ta_kwargs["lr_scheduler_type"] = "cosine"
    if "evaluation_strategy" in signature(TrainingArguments).parameters:
        ta_kwargs["evaluation_strategy"] = "no"
    if "save_strategy" in signature(TrainingArguments).parameters:
        ta_kwargs["save_strategy"] = "steps"

    training_args = TrainingArguments(**ta_kwargs)

    # sanity: report trainables
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_train/1e6:.2f}M")

    # progressive A-bits warmup: A8 -> A4
    callbacks = [ActivationBitsWarmup(switch_step=max(steps // 4, 1000), final_a_bits=4)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    if save_dir:
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
    return model


# ---------- GQA checkpoint loader (patch shapes first, then load) ----------
import os, glob
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file as safe_load_file

def load_gqa_base(base_dir: str, groups: int, dtype=torch.float32):
    """
    Load a GQA-patched OPT checkpoint saved under base_dir.
    We instantiate from config, patch to GQA, then load weights.
    """
    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) build model from config (no weights yet)
    config = AutoConfig.from_pretrained(base_dir)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    # 3) patch to GQA (use groups from arg or config if present)
    g = getattr(config, "num_key_value_heads", None)
    target_groups = groups if g is None else g
    model = patch_opt_to_gqa(model, groups=target_groups)

    # 4) load weights (safetensors shards preferred)
    weight_files = sorted(glob.glob(os.path.join(base_dir, "*.safetensors")))
    if weight_files:
        state = {}
        for wf in weight_files:
            shard = safe_load_file(wf, device="cpu")
            state.update(shard)
    else:
        ptbin = os.path.join(base_dir, "pytorch_model.bin")
        if not os.path.exists(ptbin):
            raise FileNotFoundError(
                f"No weights found in {base_dir} (no *.safetensors or pytorch_model.bin)."
            )
        state = torch.load(ptbin, map_location="cpu")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load_gqa_base] loaded with missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  missing:", sorted(missing)[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("  unexpected:", sorted(unexpected)[:10], "..." if len(unexpected) > 10 else "")

    return model, tokenizer


# =============== Orchestration (the exact order) ===============
def main():
    import argparse
    device = get_device(None)  # default, may be overwritten by CLI

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--base_dir", default="./opt125m-gqa3-up", help="Uptrained GQA=3 base directory")
    ap.add_argument("--gqa_groups", type=int, default=3, help="Target GQA groups if patching is needed")
    ap.add_argument("--alpha", type=float, default=0.6, help="SmoothQuant alpha (0..1)")
    ap.add_argument("--calib_batches", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-5)      # conservative
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)   # effective batch 64
    ap.add_argument("--save_dir", default="./opt125m-gqa3up-sq-loraqat-w4a4-stable")
    ap.add_argument("--eval_ppl", action="store_true")
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--eval_max_batches", type=int, default=50)
    args = ap.parse_args()

    if args.device:
        device = get_device(args.device)

    # 1) Load the uptrained base (GQA checkpoint) and build model shapes to match before loading
    print(f"Loading base from {args.base_dir}")
    model, tokenizer = load_gqa_base(args.base_dir, groups=args.gqa_groups, dtype=torch.float32)

    # 2) Ensure GQA (skip if already present in config)
    if is_model_gqa(model):
        print(f"Model already GQA (num_key_value_heads={getattr(model.config, 'num_key_value_heads', 'unknown')}). Skipping patch.")
    else:
        print(f"Patching OPT to GQA (groups={args.gqa_groups})...")
        model = patch_opt_to_gqa(model, groups=args.gqa_groups)

    # 3) SmoothQuant calibration -> PTQ W4A4
    print("Collecting activation stats for SmoothQuant...")
    ds = load_wikitext2()
    act_max = collect_act_max_per_input_channel(
        model, tokenizer, ds, device=device, num_batches=args.calib_batches
    )
    smooth_scales = compute_smooth_scales(model, act_max, alpha=args.alpha)
    print("Applying QuantLinear W4A4 with SmoothQuant scales...")
    replace_linear_with_quant(model, w_bits=4, a_bits=4, smooth_scales=smooth_scales)

    # 4) Inject LoRA adapters and run QAT (adapters only, W4A4 fake-quant inside LoRA)
    print("Injecting LoRA adapters and running QAT...")
    # Attention-only first for stability; start with A8 in adapters
    inject_lora(
        model, r=4, lora_alpha=16, lora_dropout=0.1, a_bits=8, w_bits=4,
        targets=("q_proj","v_proj","out_proj")
    )

    # Freeze all params, unfreeze only LoRA A/B weights
    for p in model.parameters():
        p.requires_grad = False
    lora_params = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.weight.requires_grad = True
            m.B.weight.requires_grad = True
            lora_params += m.A.weight.numel() + m.B.weight.numel()
    print(f"Trainable parameters: {lora_params/1e6:.2f}M")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    train_lora_qat(
        model, tokenizer, ds, device,
        steps=args.steps, lr=args.lr, batch=args.batch_size,
        grad_accum=args.grad_accum, save_dir=args.save_dir
    )

    # 5) Optional PPL eval
    if args.eval_ppl:
        tokenized = tokenize_dataset(ds[0], ds[1], tokenizer, block_size=1024)
        ppl = eval_ppl(
            model, tokenizer, tokenized,
            device=device, batch_size=args.eval_bs, max_batches=args.eval_max_batches
        )
        print(f"Perplexity (approx): {ppl:.3f}")
if __name__ == "__main__":
    main()
