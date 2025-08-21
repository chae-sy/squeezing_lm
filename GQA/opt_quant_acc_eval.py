import os, math, argparse
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

# =======================
# LAMBADA evaluator
# =======================
class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        # tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"])
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        dataset.set_format(type="torch", columns=["input_ids"])
        self.dataset = dataset

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, hit = 0, 0
        for ex in self.dataset:
            # single-sample forward; we avoid batching here to keep it identical to the user's code
            input_ids = ex["input_ids"].to(self.device).unsqueeze(0)
            if input_ids.size(1) < 2:
                continue  # need at least 2 tokens to predict last one
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]  # predict the last token from previous position
            pred = last_token_logits.argmax(dim=-1)
            label = input_ids[:, -1]
            total += 1
            hit += (pred == label).sum().item()
        return hit / max(1, total)

# =======================
# Quant helpers (minimal)
# =======================
def _qmax(bits: int) -> int:
    assert 2 <= bits <= 8
    return (1 << (bits - 1)) - 1

def _percentile(x: torch.Tensor, q: float | None, dim=None, keepdim=False):
    if q is None:
        return x.abs().amax(dim=dim, keepdim=keepdim)
    return torch.quantile(x.abs(), q, dim=dim, keepdim=keepdim)

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
    dev = w.device
    if group_size <= 0 or bits > 4:
        w_int, w_scale = quantize_per_channel_nbit(w, axis=1, bits=bits, clip_q=clip_q, eps=eps)
        return w_int.float() * w_scale.unsqueeze(1), {"kind":"per_channel","scale":w_scale}
    qmax = float(_qmax(bits))
    num_groups = (In + group_size - 1) // group_size
    w_q = torch.empty((out, In), dtype=torch.int8, device=dev)
    scales = torch.empty((out, num_groups), dtype=torch.float32, device=dev)
    for g in range(num_groups):
        s, e = g * group_size, min(In, (g+1) * group_size)
        w_slice = w[:, s:e]
        max_abs = _percentile(w_slice, clip_q, dim=1, keepdim=True).clamp(min=eps)
        scales[:, g] = (max_abs.squeeze(1) / qmax).to(torch.float32)
        w_q[:, s:e] = torch.round((w_slice / max_abs).clamp(-1, 1) * qmax).to(torch.int8)
    rep = torch.repeat_interleave(scales, repeats=group_size, dim=1)[:, :In]
    return w_q.float() * rep, {"kind":"group","scale":scales,"group_size":group_size}

class ActFakeQuantPerTensorFixed(nn.Module):
    def __init__(self, a_bits: int, a_scale: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.a_bits = a_bits; self.qmax = float(_qmax(a_bits)); self.eps = eps
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
        self.a_bits = a_bits; self.qmax = float(_qmax(a_bits)); self.eps = eps
        a_scale_vec = torch.clamp(a_scale_vec.to(torch.float32), min=eps)
        self.register_buffer("a_scale_vec", a_scale_vec, persistent=False)
    def update_scale_(self, new_scale_vec: torch.Tensor):
        new_scale_vec = torch.clamp(new_scale_vec.to(self.a_scale_vec.device, dtype=self.a_scale_vec.dtype), min=self.eps)
        self.a_scale_vec.data = torch.maximum(self.a_scale_vec, new_scale_vec)
    def forward(self, x: torch.Tensor):
        a = self.a_scale_vec
        while a.dim() < x.dim(): a = a.unsqueeze(0)
        a = torch.clamp(a, min=self.eps)
        with torch.no_grad():
            x_q = torch.round((x / a).clamp(-self.qmax, self.qmax))
            return x_q * a

# Plain W/A fake-quant (weights only here)
class QuantLinearNbit(nn.Module):
    def __init__(self, lin: nn.Linear, w_bits: int = 8):
        super().__init__()
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None
        w = lin.weight.detach().to(torch.float32)
        w_int, w_scale = quantize_per_channel_nbit(w, axis=1, bits=w_bits)
        self.register_buffer("w_deq", (w_int.float() * w_scale.unsqueeze(1)), persistent=False)
    def forward(self, x: torch.Tensor):
        y = torch.matmul(x, self.w_deq.t())
        if self.bias is not None: y = y + self.bias
        return y

def replace_linear_with_quant(module: nn.Module, w_bits: int = 8):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and "lm_head" not in name.lower():
            setattr(module, name, QuantLinearNbit(child, w_bits=w_bits))
        else:
            replace_linear_with_quant(child, w_bits=w_bits)

# SmoothQuant
class SQQuantLinearNbit(nn.Module):
    def __init__(self, lin: nn.Linear, s_vec: torch.Tensor, amax_vec: torch.Tensor,
                 w_bits: int = 8, a_bits: int = 8, act_channelwise: bool = False,
                 w_group_size: int = 0, w_clip_q: float | None = None,
                 bias_correction: bool = True, mean_vec: torch.Tensor | None = None):
        super().__init__()
        dev = lin.weight.device
        self.register_buffer("s_vec", s_vec.to(dev, dtype=torch.float32), persistent=False)
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None
        # W' = W * s (colwise), quantize
        w = lin.weight.detach().to(torch.float32)
        w_prime = w * (self.s_vec.unsqueeze(0))
        w_deq, _ = quantize_weight_groupwise_nbit(w_prime, bits=w_bits, group_size=w_group_size, clip_q=w_clip_q)
        self.register_buffer("w_deq", w_deq, persistent=False)
        # A-quant: Identity for A8; fixed for <8
        if a_bits >= 8:
            self.act_q = nn.Identity()
        else:
            if act_channelwise:
                a_scale_vec = (amax_vec.to(dev, dtype=torch.float32) / self.s_vec) / float(_qmax(a_bits))
                self.act_q = ActFakeQuantPerChannelNbit(a_bits, a_scale_vec)
            else:
                amax_prime_scalar = (amax_vec.to(dev, dtype=torch.float32) / self.s_vec).amax()
                a_scale = amax_prime_scalar / float(_qmax(a_bits))
                self.act_q = ActFakeQuantPerTensorFixed(a_bits, a_scale)
        # bias corr
        if bias_correction and self.bias is not None and mean_vec is not None:
            mean_x_prime = mean_vec.to(dev, dtype=torch.float32) / self.s_vec
            delta = (w_prime - w_deq)
            with torch.no_grad():
                corr = torch.matmul(delta, mean_x_prime)
                self.bias.data.add_(-corr.to(self.bias.device))

    def forward(self, x: torch.Tensor):
        s = self.s_vec
        while s.dim() < x.dim(): s = s.unsqueeze(0)
        x = x / s
        x_q = self.act_q(x) if not isinstance(self.act_q, nn.Identity) else x
        y = torch.matmul(x_q, self.w_deq.t())
        if self.bias is not None: y = y + self.bias
        return y

@torch.no_grad()
def collect_activation_amax_per_linear(model: nn.Module, calib_loader, device, max_batches=32, quantile=None):
    model.eval().to(device)
    lin_modules, amax_dict, sum_dict, count_dict = [], {}, {}, {}
    for m in model.modules():
        if isinstance(m, nn.Linear):
            lin_modules.append(m)
            amax_dict[m] = torch.zeros(m.in_features, dtype=torch.float32, device=device)
            sum_dict[m]  = torch.zeros(m.in_features, dtype=torch.float32, device=device)
            count_dict[m] = 0

    hooks = []
    def make_hook(m):
        def hook(_module, inputs):
            x = inputs[0].detach()
            x_abs = x.abs()
            if quantile is None:
                cur = x_abs.amax(dim=tuple(range(x_abs.ndim-1)))
            else:
                x_flat = x_abs.reshape(-1, x_abs.shape[-1])
                cur = torch.quantile(x_flat, quantile, dim=0)
            amax_dict[m].copy_(torch.maximum(amax_dict[m], cur))
            sum_dict[m].add_(x.sum(dim=tuple(range(x.ndim-1))))
            count_dict[m] += x.numel() // x.shape[-1]
        return hook

    for m in lin_modules:
        hooks.append(m.register_forward_pre_hook(make_hook(m), with_kwargs=False))

    for i, batch in enumerate(calib_loader):
        if max_batches and i >= max_batches: break
        _ = model(input_ids=batch["input_ids"].to(device))

    for h in hooks: h.remove()
    stats = {}
    for m in lin_modules:
        stats[m] = {
            "amax": amax_dict[m].detach().cpu(),
            "mean": (sum_dict[m] / max(1, count_dict[m])).detach().cpu(),
        }
    return stats

def apply_smoothquant_and_quantize_bits(
    model: nn.Module, act_stat: dict, w_bits: int = 8, a_bits: int = 8, alpha: float = 0.5, eps: float = 1e-8,
    act_channelwise: bool = False, w_group_size: int = 0, w_clip_q: float | None = None, bias_correction: bool = True
):
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
                s = torch.clamp(s, min=1e-3, max=1e3); s = s / s.median()
                setattr(module, name, SQQuantLinearNbit(
                    child, s, amax_vec, w_bits=w_bits, a_bits=a_bits,
                    act_channelwise=act_channelwise, w_group_size=w_group_size,
                    w_clip_q=w_clip_q, bias_correction=bias_correction, mean_vec=mean_vec
                ))
            else:
                transform(child)
    transform(model)

@torch.no_grad()
def recalibrate_act_scales_post_sq(model: nn.Module, calib_loader, device, quantile: float | None):
    model.eval().to(device)
    sq_layers = [m for m in model.modules() if isinstance(m, SQQuantLinearNbit)]
    hooks = []
    def make_hook(m: SQQuantLinearNbit):
        def hook(_module, inputs):
            if isinstance(m.act_q, nn.Identity): return
            x = inputs[0]
            s = m.s_vec
            while s.dim() < x.dim(): s = s.unsqueeze(0)
            x_abs = (x.to(s.device) / s).abs()
            if isinstance(m.act_q, ActFakeQuantPerTensorFixed):
                if quantile is None:
                    amax = x_abs.amax()
                else:
                    token_max = x_abs.view(-1, x_abs.shape[-1]).amax(dim=1)
                    try:
                        amax = torch.quantile(token_max, quantile)
                    except RuntimeError:
                        amax = torch.quantile(token_max.float().cpu(), quantile).to(token_max.device)
                a_scale = torch.clamp(amax, min=m.act_q.eps) / float(_qmax(m.act_q.a_bits))
                m.act_q.update_scale_(a_scale)
            else:
                x_flat = x_abs.view(-1, x_abs.shape[-1])
                if quantile is None:
                    amax_vec = x_flat.amax(dim=0)
                else:
                    try:
                        amax_vec = torch.quantile(x_flat, quantile, dim=0)
                    except RuntimeError:
                        amax_vec = torch.quantile(x_flat.float().cpu(), quantile, dim=0).to(x_flat.device)
                a_scale_vec = torch.clamp(amax_vec, min=m.act_q.eps) / float(_qmax(m.act_q.a_bits))
                m.act_q.update_scale_(a_scale_vec)
        return hook
    for m in sq_layers:
        hooks.append(m.register_forward_pre_hook(make_hook(m), with_kwargs=False))
    for batch in calib_loader:
        _ = model(input_ids=batch["input_ids"].to(device))
    for h in hooks: h.remove()

# =======================
# Collate for variable length (LAMBADA)
# =======================
def pad_collate(tokenizer):
    pad_id = tokenizer.pad_token_id
    def collate(examples):
        ids = [e["input_ids"] for e in examples]
        maxlen = max(int(x.size(0)) for x in ids)
        out = torch.full((len(ids), maxlen), pad_id, dtype=torch.long)
        for i, x in enumerate(ids):
            out[i, :x.size(0)] = x
        return {"input_ids": out}
    return collate

# =======================
# Main
# =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "w8a8", "sq"], required=True)
    ap.add_argument("--model_name", default="facebook/opt-125m")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lambada_split", default="validation[:1000]", help="HF split string")
    ap.add_argument("--eval_bs", type=int, default=8)

    # SQ params
    ap.add_argument("--w_bits", type=int, default=8)
    ap.add_argument("--a_bits", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--calib_batches", type=int, default=64)
    ap.add_argument("--quantile", type=float, default=0.999)
    ap.add_argument("--w_group_size", type=int, default=0)
    ap.add_argument("--w_clip_q", type=float, default=None)
    ap.add_argument("--bias_correction", action="store_true")
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print(f"Loading LAMBADA split: {args.lambada_split}")
    raw = load_dataset("lambada", split=args.lambada_split)

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Build calib loader (on tokenized LAMBADA) for SQ
    calib_ds = raw.map(lambda e: tok(e["text"]), batched=True, remove_columns=raw.column_names)
    calib_ds.set_format(type="torch", columns=["input_ids"])
    calib_loader = torch.utils.data.DataLoader(
        calib_ds, batch_size=args.eval_bs, shuffle=False, collate_fn=pad_collate(tok)
    )

    if args.mode == "fp16":
        print("Converting model to FP16…")
        model = model.half()

    elif args.mode == "w8a8":
        print("Applying plain W8A8 fake-quant (weights per-out-channel; A left Identity)…")
        model = model.float()
        replace_linear_with_quant(model, w_bits=8)

    elif args.mode == "sq":
        if args.w_bits <= 4 and args.w_group_size == 0:
            args.w_group_size = 64
        print(f"Calibrating activations for SmoothQuant (alpha={args.alpha}, q={args.quantile}) on LAMBADA…")
        model = model.float()
        act_stat = collect_activation_amax_per_linear(
            model, calib_loader, device=args.device, max_batches=args.calib_batches, quantile=args.quantile
        )
        print(f"Applying SmoothQuant + W{args.w_bits}A{args.a_bits} "
              f"(act_ch={args.a_bits<8}, w_group={args.w_group_size}, w_clip_q={args.w_clip_q}, "
              f"bias_corr={args.bias_correction})…")
        apply_smoothquant_and_quantize_bits(
            model, act_stat, w_bits=args.w_bits, a_bits=args.a_bits, alpha=args.alpha,
            act_channelwise=(args.a_bits < 8), w_group_size=args.w_group_size,
            w_clip_q=args.w_clip_q, bias_correction=args.bias_correction
        )
        print("Recalibrating activation scales post-SQ on LAMBADA…")
        recalibrate_act_scales_post_sq(model, calib_loader, device=args.device, quantile=args.quantile)

    else:
        raise ValueError("Unknown mode")

    model.config.pad_token_id = tok.pad_token_id

    # Build evaluator *after* any quantization transforms (uses same tokenizer)
    evaluator = Evaluator(raw, tok, args.device)
    print("Evaluating last-word accuracy on LAMBADA…")
    acc = evaluator.evaluate(model.to(args.device))
    print(f"[{args.mode.upper()}] LAMBADA last-word accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()

