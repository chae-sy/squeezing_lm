import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.no_grad()
def build_smoothquant_scales(model, alpha: float, eps: float = 1e-8):
    """Return dict: Linear -> {'w_col_max': [in](cpu), 'a_pre': None, 's_vec': None}"""
    out = {}
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w_col_max = m.weight.detach().abs().amax(dim=0).cpu().clamp(min=eps)  # CPU
            out[m] = {"w_col_max": w_col_max, "a_pre": None, "s_vec": None}
    return out
@torch.no_grad()
def collect_preSQ_act_percentiles(model, calib_loader, device, holder, alpha,
                                  max_batches=128, quantile=0.999):
    """Fill holder[m]['a_pre'] and holder[m]['s_vec'] (CPU tensors)."""
    model.eval().to(device)
    hooks = []

    def make_hook(m):
        def hook(module, inputs):
            x = inputs[0]                                   # [..., in] on device
            x_abs = x.detach().abs().reshape(-1, x.shape[-1])
            q = torch.quantile(x_abs, quantile, dim=0)     # [in], on device
            cur = holder[module]["a_pre"]
            holder[module]["a_pre"] = q if cur is None else torch.maximum(cur, q)
        return hook

    for m in holder.keys():
        hooks.append(m.register_forward_pre_hook(make_hook(m)))

    for i, batch in enumerate(calib_loader):
        if i >= max_batches: break
        _ = model(input_ids=batch["input_ids"].to(device))

    for h in hooks: h.remove()

    # Compute s_vec per layer (move w to same device as a_pre, then bring results to CPU)
    for m, d in holder.items():
        a_pre = d["a_pre"].to(device).clamp(min=1e-8)              # device
        w_col = d["w_col_max"].to(device).clamp(min=1e-8)          # device
        s = (a_pre.pow(alpha)) / (w_col.pow(1.0 - alpha))          # device
        s = torch.clamp(s, min=1e-3, max=1e3)
        s = s / s.median()
        d["a_pre"] = d["a_pre"].detach().cpu()                     # store CPU
        d["s_vec"] = s.detach().cpu()                              # store CPU

# -----------------------
# Perplexity evaluator (fixed accumulation)
# -----------------------
@torch.no_grad()
def eval_ppl(model, tokenizer, dataset, batch_size=8, max_batches=None, device="cuda"):
    model.eval().to(device)
    total_nll = 0.0
    total_tokens = 0
    loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)  # [B, T]
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss):
            continue
        batch_tokens = (labels != -100).sum().item() if (labels == -100).any() else labels.numel()
        total_nll += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens
    return math.exp(total_nll / max(1, total_tokens))

# -----------------------
# Dataset: WikiText-2 (raw), chunked into fixed-length blocks
# -----------------------
def build_wikitext2(tokenizer, block_size=1024):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(ex):
        return tokenizer(ex["text"])

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)

    def group_texts(examples):
        concat = []
        for ids in examples["input_ids"]:
            concat.extend(ids)
        total = (len(concat) // block_size) * block_size
        concat = concat[:total]
        return {"input_ids": [torch.tensor(concat[i:i+block_size], dtype=torch.long)
                              for i in range(0, total, block_size)]}

    val_grouped = tokenized["validation"].map(
        group_texts, batched=True, remove_columns=tokenized["validation"].column_names
    )
    val_grouped.set_format(type="torch", columns=["input_ids"])
    return {"validation": val_grouped}

# -----------------------
# Quant utilities (N-bit, symmetric)
# -----------------------
def _qmax(bits: int) -> int:
    assert 2 <= bits <= 8, "Supported bit-widths: 2..8"
    return (1 << (bits - 1)) - 1  # signed symmetric range [-qmax, qmax]

def quantize_per_channel_nbit(w: torch.Tensor, axis: int, bits: int, eps: float = 1e-8):
    """
    Symmetric per-channel n-bit quantization along `axis` of `w` (float32).
    Returns (int_codes:int8, scale:float32[channel]).
    """
    qmax = _qmax(bits)
    max_abs = w.abs().amax(dim=axis, keepdim=True).clamp(min=eps)
    scale = (max_abs / float(qmax)).squeeze(axis).to(torch.float32)
    w_q = torch.round((w / max_abs).clamp(-1, 1) * qmax).to(torch.int8)  # store as int8 codes
    return w_q, scale

# -----------------------
# Activation fake-quant (per-channel, calibrated)
# -----------------------
class ActFakeQuantNbitPerChannel(nn.Module):
    """Per-channel symmetric n-bit fake-quant using a provided per-input-channel amax vector."""
    def __init__(self, a_bits: int, amax_vec: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.qmax = float(_qmax(a_bits))
        self.eps = eps
        self.register_buffer("amax_vec", amax_vec.to(torch.float32), persistent=False)  # [in_features]

    def forward(self, x: torch.Tensor):
        scale = (self.amax_vec.clamp(min=self.eps) / self.qmax).to(x.dtype).to(x.device)  # [in]
        x_q = torch.round((x / scale).clamp(-self.qmax, self.qmax))
        return x_q * scale

# -----------------------
# Calibration: per-linear, per-input-channel percentile(|x|)
# -----------------------
@torch.no_grad()
def collect_postSQ_act_percentiles(model, calib_loader, device, holder,
                                   max_batches=128, quantile=0.999):
    """Measure percentile(|x/s|) using precomputed s_vec; return CPU tensors."""
    model.eval().to(device)
    post = {m: torch.zeros_like(d["s_vec"]) for m, d in holder.items()}

    hooks = []
    def make_hook(m):
        s_vec = holder[m]["s_vec"].to(device)                      # device
        def hook(module, inputs):
            x = inputs[0]                                          # [..., in]
            x_sq = (x / s_vec)                                     # post-SQ act
            x_abs = x_sq.detach().abs().reshape(-1, x_sq.shape[-1])
            q = torch.quantile(x_abs, quantile, dim=0)             # device
            # store on CPU
            post[m].copy_(torch.maximum(post[m], q.detach().cpu()))
        return hook

    for m in holder.keys():
        hooks.append(m.register_forward_pre_hook(make_hook(m)))

    for i, batch in enumerate(calib_loader):
        if i >= max_batches: break
        _ = model(input_ids=batch["input_ids"].to(device))

    for h in hooks: h.remove()
    return post

def a_bits_for_layer(parent_name: str, default_a_bits: int) -> int:
    # Keep attention activations at 8-bit; MLP can use user-selected a_bits
    lname = parent_name.lower()
    if any(tag in lname for tag in ["self_attn", "q_proj", "k_proj", "v_proj", "out_proj", "attn"]):
        return 8
    return default_a_bits

# -----------------------
# Plain W8A8 (no SQ) for reference
# -----------------------
class QuantLinear8A8(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None
        w = lin.weight.detach().to(torch.float32).cpu()                    # [out, in]
        w_int, w_scale = quantize_per_channel_nbit(w, axis=1, bits=8)     # per-output-channel
        self.register_buffer("w_int", w_int, persistent=False)
        self.register_buffer("w_scale", w_scale, persistent=False)
        self.register_buffer("w_deq", (self.w_int.float() * self.w_scale.unsqueeze(1)), persistent=False)

    def forward(self, x: torch.Tensor):
        y = torch.matmul(x, self.w_deq.t())
        if self.bias is not None: y = y + self.bias
        return y

def replace_linear_with_quant_8a8(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if any(k in name.lower() for k in ["lm_head"]):  # keep head safe
                continue
            setattr(module, name, QuantLinear8A8(child))
        else:
            replace_linear_with_quant_8a8(child)

# -----------------------
# SmoothQuant + (W_bits, A_bits)
# -----------------------
class SQQuantLinearNbit(nn.Module):
    """
    SmoothQuant + (W_bits, A_bits) fake-quant in one module.
    Direction: x' = x / s, W' = W * s (column-wise), then quantize W' per-output-channel,
    and A per-channel using calibrated post-SQ amax.
    """
    def __init__(self, lin: nn.Linear, s_vec: torch.Tensor, w_bits: int, a_bits: int, amax_vec_preSQ: torch.Tensor):
        super().__init__()
        assert isinstance(lin, nn.Linear)
        assert s_vec.ndim == 1 and s_vec.shape[0] == lin.in_features
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.register_buffer("s_vec", s_vec.to(torch.float32), persistent=False)  # [in]

        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None

        # SmoothQuant on weights: W' = W * s  (column-wise)
        w = lin.weight.detach().to(torch.float32).cpu()                              # [out, in]
        w = w * (self.s_vec.unsqueeze(0))                                            # broadcast over rows

        # Quantize W' per-output-channel (reduce over input dim)
        w_int, w_scale = quantize_per_channel_nbit(w, axis=1, bits=w_bits)
        self.register_buffer("w_int", w_int, persistent=False)
        self.register_buffer("w_scale", w_scale, persistent=False)
        self.register_buffer("w_deq", (self.w_int.float() * self.w_scale.unsqueeze(1)), persistent=False)

        # Activation amax AFTER SQ divide: a' = a / s
        amax_post = (amax_vec_preSQ.to(torch.float32) / self.s_vec.cpu()).clamp(min=1e-8)
        self.act_q = ActFakeQuantNbitPerChannel(a_bits, amax_post)

    def forward(self, x: torch.Tensor):
        x = x / self.s_vec                      # SmoothQuant activation scaling
        x_q = self.act_q(x)                     # per-channel fake-quant
        y = torch.matmul(x_q, self.w_deq.t())
        if self.bias is not None: y = y + self.bias
        return y

def apply_smoothquant_with_postcal(
    model, holder, post_act, w_bits: int, a_bits_default: int
):
    def transform(module: nn.Module, prefix=""):
        for name, child in list(module.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if any(k in name.lower() for k in ["lm_head"]):
                    continue
                s = holder[child]["s_vec"]
                a_pre = holder[child]["a_pre"]
                a_post = post_act[child]    # calibrated AFTER SQ
                use_a_bits = a_bits_for_layer(child_prefix, a_bits_default)
                # build quantized module
                qmod = SQQuantLinearNbit(child, s, w_bits=w_bits, a_bits=use_a_bits, amax_vec_preSQ=a_pre)
                # override its activation observer with post-SQ amax
                qmod.act_q = ActFakeQuantNbitPerChannel(use_a_bits, a_post)
                setattr(module, name, qmod)
            else:
                transform(child, child_prefix)
    transform(model)

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp16", "w8a8", "sq"], required=True)
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # SQ params
    parser.add_argument("--w_bits", type=int, default=8, help="Weight bits (2..8)")
    parser.add_argument("--a_bits", type=int, default=8, help="Activation bits (2..8)")
    parser.add_argument("--alpha", type=float, default=0.7, help="SmoothQuant α in [0,1]")
    parser.add_argument("--calib_batches", type=int, default=128, help="Calibration batches")
    parser.add_argument("--quantile", type=float, default=0.999, help="Percentile for activation amax (e.g., 0.999)")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building WikiText-2 (raw) validation set...")
    dataset = build_wikitext2(tokenizer, block_size=args.block_size)

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if args.mode == "fp16":
        print("Converting model to FP16…")
        model = model.half()

    elif args.mode == "w8a8":
        print("Applying plain W8A8 fake-quant to Linear layers…")
        model = model.float()
        replace_linear_with_quant_8a8(model)

    elif args.mode == "sq":
        print(f"Calibrating (two-pass) for SmoothQuant…")
        model = model.float()
        calib_loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=args.eval_bs, shuffle=False)

        # 1) Pre-SQ calibration to get a_pre and compute s_vec
        holder = build_smoothquant_scales(model, alpha=args.alpha)
        collect_preSQ_act_percentiles(model, calib_loader, device=args.device, alpha = args.alpha,
                                  holder=holder, max_batches=args.calib_batches, quantile=args.quantile)

        # 2) Post-SQ calibration: measure percentile(|x/s|)
        post_act = collect_postSQ_act_percentiles(model, calib_loader, device=args.device,
                                              holder=holder, max_batches=args.calib_batches, quantile=args.quantile)

        # 3) Apply SmoothQuant + (Wbits, Abits) with selective attention Abits
        print(f"Applying SmoothQuant + fake-quant (W{args.w_bits}A{args.a_bits}, attn A8)…")
        apply_smoothquant_with_postcal(model, holder, post_act, w_bits=args.w_bits, a_bits_default=args.a_bits)

    else:
        raise ValueError("Unknown mode")

    model.config.pad_token_id = tokenizer.pad_token_id

    print("Evaluating perplexity…")
    ppl = eval_ppl(model, tokenizer, dataset, batch_size=args.eval_bs,
                   max_batches=args.max_batches, device=args.device)
    if args.mode == "sq":
        print(f"[Mode: {args.mode} | W{args.w_bits}A{args.a_bits} | α={args.alpha} | q={args.quantile}] Perplexity: {ppl:.4f}")
    else:
        print(f"[Mode: {args.mode}] Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()

