import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import math, argparse, os
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use a specific GPU if you want (kept from your original)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

# -----------------------
# eval_ppl (fixed)
# -----------------------
@torch.no_grad()
def eval_ppl(model, tokenizer, dataset, batch_size=8, max_batches=None, device="cuda"):
    model.eval().to(device)
    total_nll = 0.0
    total_tokens = 0
    loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches: break
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss): continue
        batch_tokens = (labels != -100).sum().item() if (labels == -100).any() else labels.numel()
        total_nll += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens
    return math.exp(total_nll / max(1, total_tokens))

# -----------------------
# Data
# -----------------------
def build_wikitext2(tokenizer, block_size=1024):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(example):
        return tokenizer(example["text"])

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)

    def group_texts(examples):
        concat = []
        for ids in examples["input_ids"]:
            concat.extend(ids)
        total = (len(concat)//block_size)*block_size
        concat = concat[:total]
        return {"input_ids": [torch.tensor(concat[i:i+block_size], dtype=torch.long)
                              for i in range(0, total, block_size)]}

    val_grouped = tokenized["validation"].map(group_texts, batched=True,
                                              remove_columns=tokenized["validation"].column_names)
    val_grouped.set_format(type="torch", columns=["input_ids"])
    return {"validation": val_grouped}

# -----------------------
# Quant utils (generic n-bit)
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
    scale = (max_abs / float(qmax)).squeeze(axis).to(torch.float32)     # [w.shape[axis]]
    w_q = torch.round((w / max_abs).clamp(-1, 1) * qmax).to(torch.int8) # codes in int8
    return w_q, scale

class ActFakeQuantNbit(nn.Module):
    """Per-tensor symmetric n-bit fake-quant for activations (simple & fast; works well for A8/A6 with SQ)."""
    def __init__(self, a_bits: int, eps: float = 1e-8):
        super().__init__()
        self.a_bits = a_bits
        self.qmax = float(_qmax(a_bits))
        self.eps = eps
    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            max_abs = x.abs().amax().clamp(min=self.eps)
            a_scale = max_abs / self.qmax
            x_q = torch.round((x / a_scale).clamp(-self.qmax, self.qmax))
            return x_q * a_scale

# -----------------------
# Plain uniform W/A fake-quant (no SQ)
# -----------------------
class QuantLinearNbit(nn.Module):
    """Uniform W{w_bits}A{a_bits} fake-quant linear (per-out-channel W, per-tensor A)."""
    def __init__(self, lin: nn.Linear, w_bits: int = 8, a_bits: int = 8):
        super().__init__()
        assert isinstance(lin, nn.Linear)
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None

        w = lin.weight.detach().to(torch.float32).cpu()                   # [out, in]
        w_int, w_scale = quantize_per_channel_nbit(w, axis=1, bits=w_bits)
        self.register_buffer("w_int", w_int, persistent=False)
        self.register_buffer("w_scale", w_scale, persistent=False)
        self.register_buffer("w_deq", (self.w_int.float() * self.w_scale.unsqueeze(1)), persistent=False)

        self.act_q = ActFakeQuantNbit(a_bits)

    def forward(self, x: torch.Tensor):
        x_q = self.act_q(x)
        y = torch.matmul(x_q, self.w_deq.t())
        if self.bias is not None: y = y + self.bias
        return y

def replace_linear_with_quant(module: nn.Module, w_bits: int = 8, a_bits: int = 8):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # keep lm_head in FP32
            if "lm_head" in name.lower():
                continue
            setattr(module, name, QuantLinearNbit(child, w_bits=w_bits, a_bits=a_bits))
        else:
            replace_linear_with_quant(child, w_bits=w_bits, a_bits=a_bits)

# -----------------------
# SmoothQuant (generic bits)
# -----------------------
class SQQuantLinearNbit(nn.Module):
    """
    SmoothQuant + W{w_bits}A{a_bits} fake-quant in one module.
    Direction (correct SQ): x' = x / s, W' = W * s (column-wise), then quantize W' per-out-channel, A per-tensor.
    """
    def __init__(self, lin: nn.Linear, s_vec: torch.Tensor, w_bits: int = 8, a_bits: int = 8):
        super().__init__()
        assert isinstance(lin, nn.Linear)
        assert s_vec.ndim == 1 and s_vec.shape[0] == lin.in_features
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.register_buffer("s_vec", s_vec.to(torch.float32), persistent=False)  # [in]

        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None

        # W' = W * s (column-wise)
        w = lin.weight.detach().to(torch.float32).cpu()                # [out, in]
        w = w * (self.s_vec.unsqueeze(0))

        # Quantize W' per-output-channel
        w_int, w_scale = quantize_per_channel_nbit(w, axis=1, bits=w_bits)
        self.register_buffer("w_int", w_int, persistent=False)
        self.register_buffer("w_scale", w_scale, persistent=False)
        self.register_buffer("w_deq", (self.w_int.float() * self.w_scale.unsqueeze(1)), persistent=False)

        self.act_q = ActFakeQuantNbit(a_bits)

    def forward(self, x: torch.Tensor):
        # robust broadcast of s_vec to last dim
        s = self.s_vec
        while s.dim() < x.dim():
            s = s.unsqueeze(0)
        x = x / s
        x_q = self.act_q(x)
        y = torch.matmul(x_q, self.w_deq.t())
        if self.bias is not None: y = y + self.bias
        return y

@torch.no_grad()
def collect_activation_amax_per_linear(model: nn.Module, calib_loader, device, max_batches=32, quantile=None):
    """
    Returns dict: module -> amax_per_input_channel (shape [in_features]).
    If quantile is None: uses absolute max. Otherwise uses percentile(|x|).
    """
    model.eval().to(device)
    lin_modules = []
    amax_dict = {}

    for m in model.modules():
        if isinstance(m, nn.Linear):
            lin_modules.append(m)
            amax_dict[m] = torch.zeros(m.in_features, dtype=torch.float32, device=device)

    hooks = []
    def make_hook(m):
        def hook(module, inputs):
            x = inputs[0]  # [..., in]
            x_abs = x.detach().abs()
            if quantile is None:
                cur = x_abs.amax(dim=tuple(range(x_abs.ndim-1)))  # per-channel abs-max
            else:
                x_flat = x_abs.reshape(-1, x_abs.shape[-1])
                cur = torch.quantile(x_flat, quantile, dim=0)
            amax_dict[module].copy_(torch.maximum(amax_dict[module], cur))
        return hook

    for m in lin_modules:
        hooks.append(m.register_forward_pre_hook(make_hook(m), with_kwargs=False))

    for i, batch in enumerate(calib_loader):
        if i >= max_batches: break
        _ = model(input_ids=batch["input_ids"].to(device))

    for h in hooks:
        h.remove()

    for k in amax_dict:
        amax_dict[k] = amax_dict[k].detach().cpu()
    return amax_dict

def apply_smoothquant_and_quantize_bits(
    model: nn.Module, act_stat: dict, w_bits: int = 8, a_bits: int = 8, alpha: float = 0.5, eps: float = 1e-8
):
    """
    Replace every nn.Linear (except lm_head) with SQQuantLinearNbit(w_bits, a_bits).
    s = (a^alpha) / (w_col_max^(1-alpha)), then clamp & normalize for stability.
    """
    def transform(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                if "lm_head" in name.lower():
                    continue

                a = act_stat.get(child, None)
                if a is None:
                    s = torch.ones(child.in_features, dtype=torch.float32)
                else:
                    w_col_max = child.weight.detach().abs().amax(dim=0).cpu()      # [in]
                    a = torch.clamp(a, min=eps)
                    w_col_max = torch.clamp(w_col_max, min=eps)
                    s = (a.pow(alpha)) / (w_col_max.pow(1.0 - alpha))
                    s = torch.clamp(s, min=1e-3, max=1e3)
                    s = s / s.median()

                setattr(module, name, SQQuantLinearNbit(child, s, w_bits=w_bits, a_bits=a_bits))
            else:
                transform(child)
    transform(model)

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp16", "w8a8", "sq", "w8a8_sq"], required=True)
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Bits + SQ params
    parser.add_argument("--w_bits", type=int, default=8, help="Weight bits (2..8)")
    parser.add_argument("--a_bits", type=int, default=8, help="Activation bits (2..8)")
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant α in [0,1]")
    parser.add_argument("--calib_batches", type=int, default=32, help="Num batches for activation calibration")
    parser.add_argument("--quantile", type=float, default=None,
                        help="Optional percentile for activation amax (e.g., 0.999). If None, uses max.")
    args = parser.parse_args()

    # Back-compat alias
    if args.mode == "w8a8_sq":
        args.mode = "sq"
        args.w_bits, args.a_bits = 8, 8

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
        replace_linear_with_quant(model, w_bits=8, a_bits=8)

    elif args.mode == "sq":
        print(f"Calibrating activations for SmoothQuant (alpha={args.alpha}, quantile={args.quantile})…")
        model = model.float()
        calib_loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=args.eval_bs, shuffle=False)
        act_stat = collect_activation_amax_per_linear(
            model, calib_loader, device=args.device, max_batches=args.calib_batches, quantile=args.quantile
        )
        print(f"Applying SmoothQuant + W{args.w_bits}A{args.a_bits}…")
        apply_smoothquant_and_quantize_bits(
            model, act_stat, w_bits=args.w_bits, a_bits=args.a_bits, alpha=args.alpha
        )

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

