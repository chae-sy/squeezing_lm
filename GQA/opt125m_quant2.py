import math, argparse, os
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

# =========================
# Small helpers
# =========================
def _qmax(bits: int) -> int:
    assert 2 <= bits <= 8, "Supported bit-widths: 2..8"
    return (1 << (bits - 1)) - 1  # symmetric signed [-qmax, qmax]

def _percentile(x: torch.Tensor, q: float | None, dim=None, keepdim=False):
    if q is None:
        return x.abs().amax(dim=dim, keepdim=keepdim)
    return torch.quantile(x.abs(), q, dim=dim, keepdim=keepdim)

# =========================
# Data & Eval
# =========================
def build_wikitext2(tokenizer, block_size=1024):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(example):
        return tokenizer(example["text"])

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

@torch.no_grad()
def eval_ppl(model, tokenizer, dataset, batch_size=8, max_batches=None, device="cuda"):
    model.eval().to(device)
    total_nll, total_tokens = 0.0, 0
    loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False)
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches: break
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        if not torch.isfinite(loss): continue
        tokens = labels.numel()
        total_nll += float(loss.item()) * tokens
        total_tokens += tokens
    return math.exp(total_nll / max(1, total_tokens))

# =========================
# Quant primitives
# =========================
def quantize_per_channel_nbit(w: torch.Tensor, axis: int, bits: int, clip_q: float | None = None, eps: float = 1e-8):
    qmax = _qmax(bits)
    max_abs = _percentile(w, clip_q, dim=axis, keepdim=True).clamp(min=eps)
    scale = (max_abs / float(qmax)).squeeze(axis).to(dtype=torch.float32, device=w.device)
    w_q = torch.round((w / max_abs).clamp(-1, 1) * qmax).to(torch.int8)
    return w_q, scale

def quantize_weight_groupwise_nbit(w: torch.Tensor, bits: int, group_size: int = 64,
                                   clip_q: float | None = None, eps: float = 1e-8):
    assert w.dim() == 2  # [out, in]
    out, In = w.shape
    device = w.device
    if group_size <= 0 or bits > 4:
        w_int, w_scale = quantize_per_channel_nbit(w, axis=1, bits=bits, clip_q=clip_q, eps=eps)
        w_deq = w_int.float() * w_scale.unsqueeze(1)
        return w_deq, {"kind": "per_channel", "scale": w_scale}

    qmax = float(_qmax(bits))
    num_groups = (In + group_size - 1) // group_size
    w_q = torch.empty((out, In), dtype=torch.int8, device=device)
    scales = torch.empty((out, num_groups), dtype=torch.float32, device=device)
    for g in range(num_groups):
        s = g * group_size
        e = min(In, s + group_size)
        w_slice = w[:, s:e]
        max_abs = _percentile(w_slice, clip_q, dim=1, keepdim=True).clamp(min=eps)  # [out,1]
        scales[:, g] = (max_abs.squeeze(1) / qmax).to(torch.float32)
        w_q[:, s:e] = torch.round((w_slice / max_abs).clamp(-1, 1) * qmax).to(torch.int8)
    rep = torch.repeat_interleave(scales, repeats=group_size, dim=1)[:, :In]
    w_deq = w_q.float() * rep
    return w_deq, {"kind": "group", "scale": scales, "group_size": group_size}

class ActFakeQuantPerTensorFixed(nn.Module):
    def __init__(self, a_bits: int, a_scale: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.a_bits = a_bits
        self.qmax = float(_qmax(a_bits))
        self.eps = eps
        self.register_buffer("a_scale", torch.clamp(a_scale.to(torch.float32), min=eps), persistent=False)
    def update_scale_(self, new_scale: torch.Tensor):
        new_scale = torch.clamp(new_scale.to(self.a_scale.device, dtype=self.a_scale.dtype), min=self.eps)
        self.a_scale.data = torch.maximum(self.a_scale, new_scale)
    def forward(self, x: torch.Tensor):
        s = torch.clamp(self.a_scale, min=self.eps)
        with torch.no_grad():
            x_q = torch.round((x / s).clamp(-self.qmax, self.qmax))
            return x_q * s

class ActFakeQuantPerChannelNbit(nn.Module):
    def __init__(self, a_bits: int, a_scale_vec: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.a_bits = a_bits
        self.qmax = float(_qmax(a_bits))
        self.eps = eps
        a_scale_vec = torch.clamp(a_scale_vec.to(torch.float32), min=eps)
        self.register_buffer("a_scale_vec", a_scale_vec, persistent=False)
    def update_scale_(self, new_scale_vec: torch.Tensor):
        new_scale_vec = torch.clamp(new_scale_vec.to(self.a_scale_vec.device, dtype=self.a_scale_vec.dtype),
                                    min=self.eps)
        self.a_scale_vec.data = torch.maximum(self.a_scale_vec, new_scale_vec)
    def forward(self, x: torch.Tensor):
        a = self.a_scale_vec
        while a.dim() < x.dim(): a = a.unsqueeze(0)
        a = torch.clamp(a, min=self.eps)
        with torch.no_grad():
            x_q = torch.round((x / a).clamp(-self.qmax, self.qmax))
            return x_q * a

# =========================
# Plain W/A fake-quant (baseline)
# =========================
class QuantLinearNbit(nn.Module):
    """Uniform W{w_bits}A{a_bits} fake-quant (per-out-channel W; activations left as Identity here)."""
    def __init__(self, lin: nn.Linear, w_bits: int = 8, a_bits: int = 8):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None
        w = lin.weight.detach().to(torch.float32)
        w_int, w_scale = quantize_per_channel_nbit(w, axis=1, bits=w_bits)
        self.register_buffer("w_deq", (w_int.float() * w_scale.unsqueeze(1)), persistent=False)
        self.act_q = nn.Identity()
    def forward(self, x: torch.Tensor):
        x_q = self.act_q(x)
        y = torch.matmul(x_q, self.w_deq.t())
        if self.bias is not None: y = y + self.bias
        return y

def replace_linear_with_quant(module: nn.Module, w_bits: int = 8, a_bits: int = 8):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and "lm_head" not in name.lower():
            setattr(module, name, QuantLinearNbit(child, w_bits=w_bits, a_bits=a_bits))
        else:
            replace_linear_with_quant(child, w_bits=w_bits, a_bits=a_bits)

# =========================
# SmoothQuant (improved)
# =========================
class SQQuantLinearNbit(nn.Module):
    """
    SmoothQuant + fake-quant:
        x' = x / s_vec, W' = W * s_vec, quantize W' and A with fixed scales.
    - W: per-channel (6/8b) or group-wise (<=4b).
    - A: per-tensor fixed or per-channel fixed.
    - Bias correction: b <- b - (W' - W_deq) @ E[x'].
    """
    def __init__(self, lin: nn.Linear, s_vec: torch.Tensor, amax_vec: torch.Tensor,
                 w_bits: int = 8, a_bits: int = 8, act_channelwise: bool = False,
                 w_group_size: int = 0, w_clip_q: float | None = None,
                 bias_correction: bool = True, mean_vec: torch.Tensor | None = None):
        super().__init__()
        dev = lin.weight.device
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.register_buffer("s_vec", s_vec.to(dev, dtype=torch.float32), persistent=False)
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None

        # W' and its quantization
        w = lin.weight.detach().to(torch.float32)
        w_prime = w * (self.s_vec.unsqueeze(0))
        w_deq, _ = quantize_weight_groupwise_nbit(w_prime, bits=w_bits,
                                                  group_size=w_group_size, clip_q=w_clip_q)
        self.register_buffer("w_deq", w_deq, persistent=False)

        # Activation quant (fixed scales or Identity for A8)
        if a_bits >= 8:
        # A8 is robustly handled by SQ itself; leave activations unquantized
            self.act_q = nn.Identity()
        else:
            if act_channelwise:
                a_scale_vec = (amax_vec.to(dev, dtype=torch.float32) / self.s_vec) / float(_qmax(a_bits))
                self.act_q = ActFakeQuantPerChannelNbit(a_bits, a_scale_vec)
            else:
                amax_prime_scalar = (amax_vec.to(dev, dtype=torch.float32) / self.s_vec).amax()
                a_scale = amax_prime_scalar / float(_qmax(a_bits))
                self.act_q = ActFakeQuantPerTensorFixed(a_bits, a_scale)

        # Bias correction
        if bias_correction and self.bias is not None and mean_vec is not None:
            mean_x_prime = mean_vec.to(dev, dtype=torch.float32) / self.s_vec  # [in]
            delta = (w_prime - w_deq)                                          # [out, in]
            with torch.no_grad():
                corr = torch.matmul(delta, mean_x_prime)                        # [out]
                self.bias.data.add_(-corr.to(self.bias.device))

        # Optional sanity
        with torch.no_grad():
            if torch.allclose(self.w_deq, w_prime, atol=0.0, rtol=0.0):
                print("[warn] quantization produced identical weights — check config.")

    def forward(self, x: torch.Tensor):
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
    Returns dict[module] -> {"amax": [in_features], "mean": [in_features]} from the *FP model*.
    """
    model.eval().to(device)
    lin_modules, amax_dict, sum_dict, count_dict = [], {}, {}, {}
    for m in model.modules():
        if isinstance(m, nn.Linear):
            lin_modules.append(m)
            amax_dict[m] = torch.zeros(m.in_features, dtype=torch.float32, device=device)
            sum_dict[m] = torch.zeros(m.in_features, dtype=torch.float32, device=device)
            count_dict[m] = 0

    hooks = []
    def make_hook(m):
        def hook(_module, inputs):
            x = inputs[0]
            x_det = x.detach()
            x_abs = x_det.abs()
            if quantile is None:
                cur_amax = x_abs.amax(dim=tuple(range(x_abs.ndim-1)))
            else:
                x_flat = x_abs.reshape(-1, x_abs.shape[-1])
                cur_amax = torch.quantile(x_flat, quantile, dim=0)
            amax_dict[m].copy_(torch.maximum(amax_dict[m], cur_amax))
            x_sum = x_det.sum(dim=tuple(range(x_det.ndim-1)))
            sum_dict[m].add_(x_sum)
            count_dict[m] += x_det.numel() // x_det.shape[-1]
        return hook

    for m in lin_modules:
        hooks.append(m.register_forward_pre_hook(make_hook(m), with_kwargs=False))
    for i, batch in enumerate(calib_loader):
        if i >= max_batches: break
        _ = model(input_ids=batch["input_ids"].to(device))
    for h in hooks: h.remove()

    stats = {}
    for m in lin_modules:
        mean_vec = (sum_dict[m] / max(1, count_dict[m])).detach().cpu()
        amax_vec = amax_dict[m].detach().cpu()
        stats[m] = {"amax": amax_vec, "mean": mean_vec}
    return stats

def apply_smoothquant_and_quantize_bits(
    model: nn.Module, act_stat: dict, w_bits: int = 8, a_bits: int = 8, alpha: float = 0.5, eps: float = 1e-8,
    act_channelwise: bool = False, w_group_size: int = 0, w_clip_q: float | None = None, bias_correction: bool = True
):
    """
    Replace each nn.Linear (except lm_head) with SQQuantLinearNbit.
    s = (amax^alpha) / (w_col_max^(1-alpha)).
    """
    def transform(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and "lm_head" not in name.lower():
                dev = child.weight.device
                st = act_stat.get(child, None)
                if st is None:
                    amax_vec = torch.ones(child.in_features, dtype=torch.float32, device=dev)
                    mean_vec = torch.zeros(child.in_features, dtype=torch.float32, device=dev)
                else:
                    amax_vec = torch.clamp(st["amax"], min=eps).to(dev)
                    mean_vec = st["mean"].to(dev)
                w_col_max = child.weight.detach().abs().amax(dim=0).clamp(min=eps)
                s = (amax_vec.pow(alpha)) / (w_col_max.pow(1.0 - alpha))
                s = torch.clamp(s, min=1e-3, max=1e3)
                s = s / s.median()
                setattr(module, name, SQQuantLinearNbit(
                    child, s, amax_vec, w_bits=w_bits, a_bits=a_bits,
                    act_channelwise=act_channelwise,
                    w_group_size=w_group_size, w_clip_q=w_clip_q,
                    bias_correction=bias_correction, mean_vec=mean_vec
                ))
            else:
                transform(child)
    transform(model)

@torch.no_grad()
def recalibrate_act_scales_post_sq(model: nn.Module, calib_loader, device, quantile: float | None):
    """
    Re-estimate activation scales on the transformed model, robust to large tensors.
    """
    model.eval().to(device)
    sq_layers = [m for m in model.modules() if isinstance(m, SQQuantLinearNbit)]
    hooks = []

    def make_hook(m: SQQuantLinearNbit):
        def hook(_module, inputs):
            x = inputs[0]
            s = m.s_vec
            while s.dim() < x.dim(): s = s.unsqueeze(0)
            x_prime = (x.to(s.device) / s)
            x_abs = x_prime.abs()
            # skip Identity (A8 case)
            if isinstance(m.act_q, nn.Identity):
                return
            if isinstance(m.act_q, ActFakeQuantPerTensorFixed):
                if quantile is None:
                    amax_prime = x_abs.amax()
                else:
                    # reduce last dim first → length ~ B*T, avoids gigantic 1D quantile
                    token_max = x_abs.view(-1, x_abs.shape[-1]).amax(dim=1)   # [N_tokens]
                    try:
                        amax_prime = torch.quantile(token_max, quantile)
                    except RuntimeError:
                        amax_prime = torch.quantile(token_max.detach().float().cpu(), quantile).to(token_max.device)
                amax_prime = torch.clamp(amax_prime, min=m.act_q.eps)
                a_scale = amax_prime / float((1 << (m.act_q.a_bits - 1)) - 1)
                m.act_q.update_scale_(a_scale)

            elif isinstance(m.act_q, ActFakeQuantPerChannelNbit):
                x_flat = x_abs.view(-1, x_abs.shape[-1])   # [N, C]
                if quantile is None:
                    amax_vec = x_flat.amax(dim=0)
                else:
                    N = x_flat.shape[0]
                    # sample rows if huge, or CPU fallback if CUDA complains
                    if N > 2_000_000:
                        step = max(1, N // 500_000)
                        x_sub = x_flat[::step]
                        try:
                            amax_vec = torch.quantile(x_sub, quantile, dim=0)
                        except RuntimeError:
                            amax_vec = torch.quantile(x_sub.detach().float().cpu(), quantile, dim=0).to(x_sub.device)
                    else:
                        try:
                            amax_vec = torch.quantile(x_flat, quantile, dim=0)
                        except RuntimeError:
                            amax_vec = torch.quantile(x_flat.detach().float().cpu(), quantile, dim=0).to(x_flat.device)
                amax_vec = torch.clamp(amax_vec, min=m.act_q.eps)
                a_scale_vec = amax_vec / float((1 << (m.act_q.a_bits - 1)) - 1)
                m.act_q.update_scale_(a_scale_vec)
        return hook

    for m in sq_layers:
        hooks.append(m.register_forward_pre_hook(make_hook(m), with_kwargs=False))
    for batch in calib_loader:
        _ = model(input_ids=batch["input_ids"].to(device))
    for h in hooks: h.remove()

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp16", "w8a8", "sq", "w8a8_sq"], required=True)
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Bits + SQ params
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--calib_batches", type=int, default=32)

    parser.add_argument("--quantile", type=float, default=None,
                        help="Percentile for activation amax (e.g., 0.999 or 0.9995). If None, uses max.")
    parser.add_argument("--act_channelwise", action="store_true")
    parser.add_argument("--w_group_size", type=int, default=0)
    parser.add_argument("--w_clip_q", type=float, default=None)
    parser.add_argument("--bias_correction", action="store_true")

    args = parser.parse_args()

    if args.mode == "w8a8_sq":
        args.mode = "sq"; args.w_bits, args.a_bits = 8, 8

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
        # robust defaults
        if args.a_bits >= 8:
            args.act_channelwise = False  # we will use Identity for A8 inside the module
        else:
            if not args.act_channelwise:
                args.act_channelwise = True  # <=6b: channelwise A

        if args.w_bits <= 4 and args.w_group_size == 0:
            args.w_group_size = 64

        if args.w_clip_q is None and args.w_bits <= 4:
            args.w_clip_q = 0.9995

        print(f"Calibrating activations for SmoothQuant (alpha={args.alpha}, q={args.quantile})…")
        model = model.float()
        calib_loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=args.eval_bs, shuffle=False)
        act_stat = collect_activation_amax_per_linear(
            model, calib_loader, device=args.device, max_batches=args.calib_batches, quantile=args.quantile
        )

        print(f"Applying SmoothQuant + W{args.w_bits}A{args.a_bits} "
              f"(act_ch={args.act_channelwise}, w_group={args.w_group_size}, w_clip_q={args.w_clip_q}, "
              f"bias_corr={args.bias_correction})…")
        apply_smoothquant_and_quantize_bits(
            model, act_stat, w_bits=args.w_bits, a_bits=args.a_bits, alpha=args.alpha,
            act_channelwise=args.act_channelwise, w_group_size=args.w_group_size,
            w_clip_q=args.w_clip_q, bias_correction=args.bias_correction
        )

        print("Recalibrating activation scales on transformed model…")
        recalibrate_act_scales_post_sq(model, calib_loader, device=args.device, quantile=args.quantile)

    else:
        raise ValueError("Unknown mode")

    model.config.pad_token_id = tokenizer.pad_token_id

    print("Evaluating perplexity…")
    ppl = eval_ppl(model, tokenizer, dataset, batch_size=args.eval_bs,
                   max_batches=args.max_batches, device=args.device)
    if args.mode == "sq":
        print(f"[Mode: {args.mode} | W{args.w_bits}A{args.a_bits} | α={args.alpha} | "
              f"act_ch={args.act_channelwise} | w_group={args.w_group_size} | "
              f"w_clip_q={args.w_clip_q} | q={args.quantile} | bias_corr={args.bias_correction}] "
              f"Perplexity: {ppl:.4f}")
    else:
        print(f"[Mode: {args.mode}] Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()

