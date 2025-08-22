import math, argparse, os, glob, re
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------- LAMBADA helpers --------------------
class LambadaEvaluator:
    """
    Accuracy on LAMBADA with the task:
      - Given an input sequence, predict the last word.
      - We compare argmax(logits at position -2) to the true last token (label = input_ids[-1]).
    """
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def evaluate(self, model):
        model.eval().to(self.device)
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device)
            if input_ids.numel() < 2:
                continue
            input_ids = input_ids.unsqueeze(0)  # [1, T]
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        return hit / max(1, total)

def build_lambada(tokenizer, samples=1000):
    raw = load_dataset("lambada", split=f"validation[:{samples}]")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized = raw.map(tokenize_function, batched=True, remove_columns=raw.column_names)

    def filter_short(ex):
        return len(ex["input_ids"]) >= 2

    tokenized = tokenized.filter(filter_short)
    tokenized.set_format(type="torch", columns=["input_ids"])
    return tokenized
# ---------------------------------------------------------


def _qmax(bits: int) -> int:
    assert 2 <= bits <= 8, "Supported bit-widths: 2..8"
    return (1 << (bits - 1)) - 1  # symmetric signed [-qmax, qmax]

def _percentile(x: torch.Tensor, q: float | None, dim=None, keepdim=False):
    if q is None:
        return x.abs().amax(dim=dim, keepdim=keepdim)
    return torch.quantile(x.abs(), q, dim=dim, keepdim=keepdim)

def build_wikitext2_trainval(tokenizer, block_size=1024):
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
        return {
            "input_ids": [
                torch.tensor(concat[i:i + block_size], dtype=torch.long)
                for i in range(0, total, block_size)
            ]
        }

    train_grouped = tokenized["train"].map(
        group_texts, batched=True, remove_columns=tokenized["train"].column_names
    )
    val_grouped = tokenized["validation"].map(
        group_texts, batched=True, remove_columns=tokenized["validation"].column_names
    )
    train_grouped.set_format(type="torch", columns=["input_ids"])
    val_grouped.set_format(type="torch", columns=["input_ids"])
    return {"train": train_grouped, "validation": val_grouped}

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

def quantize_per_channel_nbit(w: torch.Tensor, axis: int, bits: int, clip_q: float | None = None, eps: float = 1e-8):
    qmax = _qmax(bits)
    max_abs = _percentile(w, clip_q, dim=axis, keepdim=True).clamp(min=eps)
    scale = (max_abs / float(qmax)).squeeze(axis).to(dtype=torch.float32, device=w.device)
    w_q = torch.round((w / max_abs).clamp(-1, 1) * qmax).to(torch.int8)
    return w_q, scale

def quantize_weight_groupwise_nbit(
    w: torch.Tensor, bits: int, group_size: int = 64, clip_q: float | None = None, eps: float = 1e-8
):
    assert w.dim() == 2
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
        max_abs = _percentile(w_slice, clip_q, dim=1, keepdim=True).clamp(min=eps)
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
        new_scale_vec = torch.clamp(new_scale_vec.to(self.a_scale_vec.device, dtype=self.a_scale_vec.dtype), min=self.eps)
        self.a_scale_vec.data = torch.maximum(self.a_scale_vec, new_scale_vec)
    def forward(self, x: torch.Tensor):
        a = self.a_scale_vec
        while a.dim() < x.dim():
            a = a.unsqueeze(0)
        a = torch.clamp(a, min=self.eps)
        with torch.no_grad():
            x_q = torch.round((x / a).clamp(-self.qmax, self.qmax))
            return x_q * a


class ActFakeQuantPerChannelQAT(nn.Module):
    def __init__(self, a_bits: int, init_scale_vec: torch.Tensor, momentum: float = 0.99,
                 quantile: float | None = 0.9995, eps: float = 1e-8):
        super().__init__()
        self.a_bits = a_bits
        self.qmax = float(_qmax(a_bits))
        self.momentum = float(momentum)
        self.quantile = quantile
        self.eps = eps
        init_scale_vec = torch.clamp(init_scale_vec.to(torch.float32), min=eps)
        self.register_buffer("a_scale_vec", init_scale_vec, persistent=False)

    def _update_scale(self, x: torch.Tensor):
        x_abs = x.detach().abs()
        x_flat = x_abs.view(-1, x_abs.shape[-1])
        if self.quantile is None:
            amax_vec = x_flat.amax(dim=0)
        else:
            try:
                amax_vec = torch.quantile(x_flat, self.quantile, dim=0)
            except RuntimeError:
                amax_vec = torch.quantile(x_flat.float().cpu(), self.quantile, dim=0).to(x_flat.device)
        new_scale = torch.clamp(amax_vec / self.qmax, min=self.eps)
        ema = self.a_scale_vec * self.momentum + new_scale * (1.0 - self.momentum)
        self.a_scale_vec.data = torch.maximum(ema, new_scale)

    def forward(self, x: torch.Tensor):
        self._update_scale(x)
        a = self.a_scale_vec
        while a.dim() < x.dim():
            a = a.unsqueeze(0)
        a = torch.clamp(a, min=self.eps)
        y = (x / a).clamp(-self.qmax, self.qmax)
        y_q = torch.round(y)
        y_q = (y_q - y).detach() + y
        return y_q * a


class QuantLinearNbit(nn.Module):
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


class SQQuantLinearNbit(nn.Module):
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

        w = lin.weight.detach().to(torch.float32)
        w_prime = w * (self.s_vec.unsqueeze(0))
        w_deq, _ = quantize_weight_groupwise_nbit(w_prime, bits=w_bits,
                                                  group_size=w_group_size, clip_q=w_clip_q)
        self.register_buffer("w_deq", w_deq, persistent=False)

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
        x_q = self.act_q(x)
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
                    act_channelwise=(a_bits < 8), w_group_size=w_group_size,
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
            x_prime = (x.to(s.device) / s)
            x_abs = x_prime.abs()

            if isinstance(m.act_q, ActFakeQuantPerTensorFixed):
                if quantile is None:
                    amax_prime = x_abs.amax()
                else:
                    token_max = x_abs.view(-1, x_abs.shape[-1]).amax(dim=1)
                    try:
                        amax_prime = torch.quantile(token_max, quantile)
                    except RuntimeError:
                        amax_prime = torch.quantile(token_max.detach().float().cpu(), quantile).to(token_max.device)
                amax_prime = torch.clamp(amax_prime, min=m.act_q.eps)
                a_scale = amax_prime / float(_qmax(m.act_q.a_bits))
                m.act_q.update_scale_(a_scale)

            elif isinstance(m.act_q, ActFakeQuantPerChannelNbit):
                x_flat = x_abs.view(-1, x_abs.shape[-1])
                if quantile is None:
                    amax_vec = x_flat.amax(dim=0)
                else:
                    N = x_flat.shape[0]
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
                a_scale_vec = amax_vec / float(_qmax(m.act_q.a_bits))
                m.act_q.update_scale_(a_scale_vec)
        return hook
    for m in sq_layers:
        hooks.append(m.register_forward_pre_hook(make_hook(m), with_kwargs=False))
    for batch in calib_loader:
        _ = model(input_ids=batch["input_ids"].to(device))
    for h in hooks: h.remove()

class SQQuantLoRALinearW4A4(nn.Module):
    def __init__(
        self,
        lin: nn.Linear,
        s_vec: torch.Tensor,
        amax_vec: torch.Tensor,
        *,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        w_clip_q: float = 0.9995,
        group_size: int = 64,
        bias_correction: bool = True,
        mean_vec: torch.Tensor | None = None,
        train_bias: bool = False,
        eps: float = 1e-8,
        qat_momentum: float = 0.99,
        qat_quantile: float | None = 0.9995,
    ):
        super().__init__()
        dev = lin.weight.device

        self.in_features  = lin.in_features
        self.out_features = lin.out_features
        self.rank         = int(rank)
        self.scaling      = float(lora_alpha) / max(1, self.rank)

        self.register_buffer("s_vec", s_vec.to(dev, dtype=torch.float32), persistent=False)

        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.detach().clone())
            self.bias.requires_grad_(bool(train_bias))
        else:
            self.bias = None

        w = lin.weight.detach().to(torch.float32)
        w_prime = w * (self.s_vec.unsqueeze(0))
        w_deq, _ = quantize_weight_groupwise_nbit(
            w_prime, bits=4, group_size=group_size, clip_q=w_clip_q, eps=eps
        )
        self.register_buffer("w_deq", w_deq, persistent=False)

        a_init = (amax_vec.to(dev, dtype=torch.float32) / self.s_vec) / float(_qmax(4))
        a_init = torch.clamp(a_init, min=eps)
        self.act_q  = ActFakeQuantPerChannelQAT(4, a_init, momentum=qat_momentum, quantile=qat_quantile, eps=eps)

        self.lora_A = nn.Parameter(torch.empty(self.in_features, self.rank, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features, dtype=torch.float32, device=dev))
        nn.init.normal_(self.lora_A, mean=0.0, std=1e-3)
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if bias_correction and (self.bias is not None) and (mean_vec is not None):
            mean_x_prime = mean_vec.to(dev, dtype=torch.float32) / self.s_vec
            delta = (w_prime - w_deq)
            with torch.no_grad():
                corr = torch.matmul(delta, mean_x_prime)
                self.bias.data.add_(-corr.to(self.bias.device))

        for p in self.parameters():
            p.requires_grad_(False)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(bool(train_bias))

        assert self.lora_A.shape == (self.in_features, self.rank)
        assert self.lora_B.shape == (self.rank, self.out_features)

    def forward(self, x: torch.Tensor):
        s = self.s_vec
        while s.dim() < x.dim():
            s = s.unsqueeze(0)
        x = x / s

        x_q = self.act_q(x)
        y = torch.matmul(x_q, self.w_deq.t())

        xq_drop = self.lora_dropout(x_q)
        r = torch.matmul(xq_drop, self.lora_A)
        delta = torch.matmul(r, self.lora_B)
        y = y + delta * self.scaling

        if self.bias is not None:
            y = y + self.bias
        return y

    def lora_parameters(self):
        params = [self.lora_A, self.lora_B]
        if self.bias is not None and self.bias.requires_grad:
            params.append(self.bias)
        return params

    @torch.no_grad()
    def merge_lora_(self):
        W_delta = torch.matmul(self.lora_A, self.lora_B) * self.scaling
        self.w_deq.add_(W_delta.t())

    def set_lora_scale(self, new_scale: float):
        self.scaling = float(new_scale)

class SQQuantLoRALinearOnSQBase(nn.Module):
    def __init__(
        self,
        lin: nn.Linear,
        s_vec: torch.Tensor,
        amax_vec: torch.Tensor,
        *,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias_correction: bool = True,
        mean_vec: torch.Tensor | None = None,
        eps: float = 1e-8,
        qat_momentum: float = 0.99,
        qat_quantile: float | None = 0.9995,
    ):
        super().__init__()
        dev = lin.weight.device
        self.in_features  = lin.in_features
        self.out_features = lin.out_features
        self.rank         = int(rank)
        self.scaling      = float(lora_alpha) / max(1, self.rank)

        self.register_buffer("s_vec", torch.clamp(s_vec.to(dev, dtype=torch.float32), min=eps), persistent=False)

        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.detach().clone())
        else:
            self.bias = None

        w = lin.weight.detach().to(torch.float32)
        w_int8, w_scale = quantize_per_channel_nbit(w * self.s_vec.unsqueeze(0), axis=1, bits=8)
        w_deq = w_int8.float() * w_scale.unsqueeze(1)
        self.register_buffer("w_deq", w_deq, persistent=False)

        a_init = (torch.clamp(amax_vec.to(dev, dtype=torch.float32), min=eps) / self.s_vec) / float(_qmax(4))
        a_init = torch.clamp(a_init, min=eps)
        self.act_q = ActFakeQuantPerChannelQAT(4, a_init, momentum=qat_momentum, quantile=qat_quantile, eps=eps)

        self.lora_A = nn.Parameter(torch.empty(self.in_features, self.rank, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features, dtype=torch.float32, device=dev))
        nn.init.normal_(self.lora_A, mean=0.0, std=1e-3)
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if bias_correction and (self.bias is not None) and (mean_vec is not None):
            with torch.no_grad():
                mean_x_prime = mean_vec.to(dev, dtype=torch.float32) / self.s_vec
                delta = (w * self.s_vec.unsqueeze(0) - self.w_deq)
                corr = torch.matmul(delta, mean_x_prime)
                self.bias.data.add_(-corr.to(self.bias.device))

        for p in self.parameters():
            p.requires_grad_(False)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        s = self.s_vec
        while s.dim() < x.dim(): s = s.unsqueeze(0)
        x_prime = x / s

        y = torch.matmul(x_prime, self.w_deq.t())

        xq = self.act_q(x_prime)
        r = torch.matmul(self.lora_dropout(xq), self.lora_A)
        delta = torch.matmul(r, self.lora_B)
        y = y + delta * self.scaling

        if self.bias is not None:
            y = y + self.bias
        return y

    def lora_parameters(self):
        return [self.lora_A, self.lora_B]

    @torch.no_grad()
    def merge_lora_(self):
        W_delta = torch.matmul(self.lora_A, self.lora_B) * self.scaling
        self.w_deq.add_(W_delta.t())

def apply_smoothquant_with_lora_on_sq_base(
    model: nn.Module,
    act_stat: dict,
    *,
    alpha: float = 0.80,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    qat_momentum: float = 0.99,
    qat_quantile: float | None = 0.9995,
    eps: float = 1e-8,
):
    trainable = []
    def transform(module: nn.Module):
        nonlocal trainable
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

                layer = SQQuantLoRALinearOnSQBase(
                    child, s, amax_vec,
                    rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                    bias_correction=True, mean_vec=mean_vec,
                    eps=eps, qat_momentum=qat_momentum, qat_quantile=qat_quantile
                )
                setattr(module, name, layer)
                trainable += layer.lora_parameters()
            else:
                transform(child)
    transform(model)
    return trainable

@torch.no_grad()
def recalibrate_act_scales_post_lora_on_sq_base(model: nn.Module, calib_loader, device, quantile: float | None):
    model.eval().to(device)
    layers = [m for m in model.modules() if isinstance(m, SQQuantLoRALinearOnSQBase)]
    hooks = []
    def make_hook(m: SQQuantLoRALinearOnSQBase):
        def hook(_module, inputs):
            x = inputs[0]
            s = m.s_vec
            while s.dim() < x.dim(): s = s.unsqueeze(0)
            x_prime = (x.to(s.device) / s)
            x_abs = x_prime.abs()
            x_flat = x_abs.view(-1, x_abs.shape[-1])
            if quantile is None:
                amax_vec = x_flat.amax(dim=0)
            else:
                N = x_flat.shape[0]
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
            a_scale_vec = torch.clamp(amax_vec, min=m.act_q.eps) / float(_qmax(4))
            m.act_q.a_scale_vec.data = torch.maximum(m.act_q.a_scale_vec, a_scale_vec)
        return hook
    for m in layers:
        hooks.append(m.register_forward_pre_hook(make_hook(m), with_kwargs=False))
    for batch in calib_loader:
        _ = model(input_ids=batch["input_ids"].to(device))
    for h in hooks: h.remove()

def apply_smoothquant_with_lora_w4a4(model: nn.Module, act_stat: dict, alpha: float = 0.80,
                                     w_clip_q: float = 0.9995, group_size: int = 64,
                                     lora_rank: int = 8, lora_alpha: float = 16.0,
                                     lora_dropout: float = 0.0,
                                     qat_momentum: float = 0.99,
                                     qat_quantile: float | None = 0.9995,
                                     eps: float = 1e-8):
    trainable = []
    def transform(module: nn.Module):
        nonlocal trainable
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

                lora_layer = SQQuantLoRALinearW4A4(
                    child, s, amax_vec,
                    rank=lora_rank, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    w_clip_q=w_clip_q, group_size=group_size,
                    bias_correction=True, mean_vec=mean_vec,
                    train_bias=False, eps=eps,
                    qat_momentum=qat_momentum, qat_quantile=qat_quantile
                )
                setattr(module, name, lora_layer)
                trainable += lora_layer.lora_parameters()
            else:
                transform(child)
    transform(model)
    return trainable

@torch.no_grad()
def recalibrate_act_scales_post_lora_w4a4(model: nn.Module, calib_loader, device, quantile: float | None):
    model.eval().to(device)
    lora_layers = [m for m in model.modules() if isinstance(m, SQQuantLoRALinearW4A4)]
    hooks = []
    def make_hook(m: SQQuantLoRALinearW4A4):
        def hook(_module, inputs):
            x = inputs[0]
            s = m.s_vec
            while s.dim() < x.dim(): s = s.unsqueeze(0)
            x_prime = (x.to(s.device) / s)
            x_abs = x_prime.abs()
            x_flat = x_abs.view(-1, x_abs.shape[-1])
            if quantile is None:
                amax_vec = x_flat.amax(dim=0)
            else:
                N = x_flat.shape[0]
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
            a_scale_vec = torch.clamp(amax_vec, min=m.act_q.eps) / float(_qmax(4))
            m.act_q.a_scale_vec.data = torch.maximum(m.act_q.a_scale_vec, a_scale_vec)
        return hook
    for m in lora_layers:
        hooks.append(m.register_forward_pre_hook(make_hook(m), with_kwargs=False))
    for batch in calib_loader:
        _ = model(input_ids=batch["input_ids"].to(device))
    for h in hooks: h.remove()


def lora_qat_train(model, train_loader, device, lr=3e-4, weight_decay=0.0,
                   steps=1000, grad_accum=1, warmup_steps=100, print_every=50,
                   max_grad_norm=1.0):
    model.train().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-6, step / max(1, warmup_steps))
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    step = 0
    running = 0.0
    it = iter(train_loader)
    while step < steps:
        opt.zero_grad()
        for _ in range(grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()
            running += loss.item()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        opt.step(); sched.step()
        step += 1
        if step % print_every == 0:
            print(f"[QAT] step {step}/{steps} | loss={running/print_every:.4f} | lr={opt.param_groups[0]['lr']:.2e}")
            running = 0.0

# -------------------- GQA Uptrained params loader --------------------
def _strip_prefixes(sd):
    """Normalize common prefixes like 'module.' or 'model.' to improve key matching."""
    out = {}
    for k, v in sd.items():
        nk = k
        nk = nk[7:] if nk.startswith("module.") else nk
        nk = nk[6:] if nk.startswith("model.") else nk
        out[nk] = v
    return out

def _load_state_dict_file(path):
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load
        except Exception as e:
            raise RuntimeError(f"Install safetensors to load {path}") from e
        return safe_load(path)
    else:
        return torch.load(path, map_location="cpu")

def load_uptrained_params(model: nn.Module, up_path: str):
    """
    Loads uptrained GQA params from a directory or file.
    Accepts .bin/.pt/.safetensors (single or sharded).
    Returns a (missing, unexpected) tuple from load_state_dict.
    """
    if not up_path:
        print("[UP] No up_dir provided; skipping.")
        return None

    files = []
    if os.path.isfile(up_path):
        files = [up_path]
    elif os.path.isdir(up_path):
        # Prefer sharded order if present
        shard_pat = re.compile(r".*-(\d+)-of-(\d+)\.(bin|safetensors)$")
        all_files = glob.glob(os.path.join(up_path, "*.safetensors")) + \
                    glob.glob(os.path.join(up_path, "*.bin")) + \
                    glob.glob(os.path.join(up_path, "*.pt"))
        # Sort shards numerically if matched
        shards = sorted([f for f in all_files if shard_pat.match(f)],
                        key=lambda f: int(shard_pat.match(f).group(1)))
        if shards:
            files = shards
        else:
            # Fall back to common single-file names first
            priority = ["pytorch_model.bin", "model.safetensors", "adapter_model.bin", "state_dict.pt"]
            for p in priority:
                cand = os.path.join(up_path, p)
                if os.path.exists(cand):
                    files = [cand]; break
            if not files:
                files = sorted(all_files)
    else:
        print(f"[UP] up_dir '{up_path}' not found; skipping.")
        return None

    if not files:
        print(f"[UP] No loadable files found in '{up_path}'; skipping.")
        return None

    print(f"[UP] Loading uptrained params from {len(files)} file(s):")
    for f in files: print("     -", f)

    merged = {}
    loaded_files = 0
    for f in files:
        try:
            sd = _load_state_dict_file(f)
            sd = _strip_prefixes(sd)
            merged.update(sd)
            loaded_files += 1
        except Exception as e:
            print(f"[UP] Warning: failed to read {f}: {e}")

    if loaded_files == 0:
        print("[UP] Could not load any state dict file; skipping.")
        return None

    # Try loading non-strict (GQA shapes may not fully match base arch)
    result = model.load_state_dict(merged, strict=False)
    try:
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
    except Exception:
        missing, unexpected = [], []
    print(f"[UP] Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        print("     e.g.,", missing[:8])
    if unexpected:
        print("     e.g.,", unexpected[:8])
    return (missing, unexpected)
# --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["fp16", "w8a8", "sq", "w8a8_sq", "lora_qat_w4a4", "lora_qat_w4a4_sq"],
        required=True)
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--up_dir", type=str, default="./opt125m-gqa3-up",
                        help="Directory or file containing uptrained GQA params to load before quantization.")
    parser.add_argument("--skip_up", action="store_true",
                        help="If set, do not load uptrained params.")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # SQ/quant params
    parser.add_argument("--w_bits", type=int, default=8)
    parser.add_argument("--a_bits", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--calib_batches", type=int, default=32)
    parser.add_argument("--quantile", type=float, default=0.999, help="Percentile for activation amax (None=max).")
    parser.add_argument("--w_clip_q", type=float, default=None)
    parser.add_argument("--w_group_size", type=int, default=0)
    parser.add_argument("--bias_correction", action="store_true")

    # LoRA-QAT specific
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--qat_steps", type=int, default=800)
    parser.add_argument("--qat_lr", type=float, default=3e-4)
    parser.add_argument("--qat_wd", type=float, default=0.0)
    parser.add_argument("--qat_warmup", type=int, default=100)
    parser.add_argument("--qat_accum", type=int, default=1)
    parser.add_argument("--train_bs", type=int, default=8)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--eval_lambada", action="store_true",
                        help="If set, also evaluate LAMBADA last-word accuracy.")
    parser.add_argument("--lambada_samples", type=int, default=1000,
                        help="Number of validation samples for LAMBADA, e.g., 1000.")

    args = parser.parse_args()

    if args.mode == "w8a8_sq":
        args.mode = "sq"; args.w_bits, args.a_bits = 8, 8

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Data: for QAT we need train+val; others val only
    if args.mode == "lora_qat_w4a4":
        print("Building WikiText-2 (raw) train & validation sets...")
        data = build_wikitext2_trainval(tokenizer, block_size=args.block_size)
    elif args.mode == "lora_qat_w4a4_sq":
        print("Building WikiText-2 (raw) train & validation sets (SQ base + LoRA QAT)…")
        data = build_wikitext2_trainval(tokenizer, block_size=args.block_size)
    else:
        print("Building WikiText-2 (raw) validation set...")
        valonly = build_wikitext2_trainval(tokenizer, block_size=args.block_size)["validation"]
        data = {"validation": valonly}

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # ----- Load uptrained GQA params BEFORE any quantization/SQ/LoRA -----
    if not args.skip_up and args.up_dir:
        print(f"[UP] Attempting to load uptrained params from: {args.up_dir}")
        _ = load_uptrained_params(model, args.up_dir)
    else:
        print("[UP] Skipping uptrained params load.")

    # Keep lm_head in FP32 (we never replace it)
    if args.mode == "fp16":
        print("Converting model to FP16…")
        model = model.half()

    elif args.mode == "w8a8":
        print("Applying plain W8A8 fake-quant to Linear layers…")
        model = model.float()
        replace_linear_with_quant(model, w_bits=8, a_bits=8)

    elif args.mode == "sq":
        if args.a_bits >= 8:
            args.act_channelwise = False
        else:
            args.act_channelwise = True
        if args.w_bits <= 4 and args.w_group_size == 0:
            args.w_group_size = 64

        print(f"Calibrating activations for SmoothQuant (alpha={args.alpha}, q={args.quantile})…")
        model = model.float()
        calib_loader = torch.utils.data.DataLoader(data["validation"], batch_size=args.eval_bs, shuffle=False)
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
        print("Recalibrating activation scales on transformed model…")
        recalibrate_act_scales_post_sq(model, calib_loader, device=args.device, quantile=args.quantile)

    elif args.mode == "lora_qat_w4a4":
        if args.w_group_size == 0: args.w_group_size = 64
        if args.w_clip_q is None: args.w_clip_q = 0.9995
        if args.alpha < 0.75: args.alpha = 0.80

        print(f"Calibrating activations for SmoothQuant (alpha={args.alpha}, q={args.quantile})…")
        model = model.float()
        calib_loader = torch.utils.data.DataLoader(data["train"], batch_size=args.eval_bs, shuffle=False)
        act_stat = collect_activation_amax_per_linear(
            model, calib_loader, device=args.device, max_batches=args.calib_batches, quantile=args.quantile
        )

        print("Injecting SQ+LoRA (QAT) with W4A4 (act_ch=True, w_group=64)…")
        trainable = apply_smoothquant_with_lora_w4a4(
            model, act_stat, alpha=args.alpha,
            w_clip_q=args.w_clip_q, group_size=args.w_group_size,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            qat_momentum=0.99, qat_quantile=args.quantile
        )

        print(f"Recalibrating A4 per-channel scales post-SQ (LoRA) …")
        recalibrate_act_scales_post_lora_w4a4(model, calib_loader, device=args.device, quantile=args.quantile)

        print(f"Starting LoRA QAT: steps={args.qat_steps}, lr={args.qat_lr}, rank={args.lora_rank}, alpha={args.lora_alpha}")
        train_loader = torch.utils.data.DataLoader(
            data["train"], batch_size=args.train_bs, shuffle=True, drop_last=True
        )
        if args.max_train_batches:
            class LimitedLoader:
                def __init__(self, dl, maxb): self.dl, self.maxb = dl, maxb
                def __iter__(self):
                    it, c = iter(self.dl), 0
                    for b in it:
                        if c >= self.maxb: break
                        yield b; c += 1
                def __len__(self): return min(len(self.dl), self.maxb)
            train_loader = LimitedLoader(train_loader, args.max_train_batches)

        for p in trainable: p.requires_grad_(True)
        lora_qat_train(
            model, train_loader, device=args.device,
            lr=args.qat_lr, weight_decay=args.qat_wd,
            steps=args.qat_steps, grad_accum=args.qat_accum,
            warmup_steps=args.qat_warmup, print_every=max(10, args.qat_steps//20),
            max_grad_norm=1.0
        )

    elif args.mode == "lora_qat_w4a4_sq":
        # Keep base as SQ W8A8, train LoRA adapters with A4 (QAT)
        if args.alpha < 0.75:
            args.alpha = 0.80

        print(f"Calibrating activations for SmoothQuant (alpha={args.alpha}, q={args.quantile})…")
        model = model.float()
        calib_loader = torch.utils.data.DataLoader(
            data["train"], batch_size=args.eval_bs, shuffle=False
        )
        act_stat = collect_activation_amax_per_linear(
            model, calib_loader, device=args.device, max_batches=args.calib_batches, quantile=args.quantile
        )

        print("Injecting SQ base (W8A8) + LoRA(QAT A4) adapters …")
        trainable = apply_smoothquant_with_lora_on_sq_base(
            model, act_stat,
            alpha=args.alpha,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            qat_momentum=0.99, qat_quantile=args.quantile
        )

        print("Recalibrating A4 per-channel scales for LoRA path …")
        recalibrate_act_scales_post_lora_on_sq_base(model, calib_loader, device=args.device, quantile=args.quantile)

        print(f"Starting LoRA QAT on SQ base: steps={args.qat_steps}, lr={args.qat_lr}, rank={args.lora_rank}, alpha={args.lora_alpha}")
        train_loader = torch.utils.data.DataLoader(
            data["train"], batch_size=args.train_bs, shuffle=True, drop_last=True
        )
        if args.max_train_batches:
            class LimitedLoader:
                def __init__(self, dl, maxb): self.dl, self.maxb = dl, maxb
                def __iter__(self):
                    it, c = iter(self.dl), 0
                    for b in it:
                        if c >= self.maxb: break
                        yield b; c += 1
                def __len__(self): return min(len(self.dl), self.maxb)
            train_loader = LimitedLoader(train_loader, args.max_train_batches)

        for p in trainable: p.requires_grad_(True)
        lora_qat_train(
            model, train_loader, device=args.device,
            lr=args.qat_lr, weight_decay=args.qat_wd,
            steps=args.qat_steps, grad_accum=args.qat_accum,
            warmup_steps=args.qat_warmup, print_every=max(10, args.qat_steps//20),
            max_grad_norm=1.0
        )

    else:
        raise ValueError("Unknown mode")

    model.config.pad_token_id = tokenizer.pad_token_id
    print("Evaluating perplexity…")
    ppl = eval_ppl(model, tokenizer, data if "validation" in data else {"validation": data["validation"]},
                   batch_size=args.eval_bs, max_batches=args.max_batches, device=args.device)
    print(f"[Mode: {args.mode}] Perplexity: {ppl:.4f}")
    if args.eval_lambada:
        print(f"Building LAMBADA validation[:{args.lambada_samples}]…")
        lambada_ds = build_lambada(tokenizer, samples=args.lambada_samples)
        print("Evaluating LAMBADA last-word accuracy…")
        lambada_eval = LambadaEvaluator(lambada_ds, tokenizer, args.device)
        lambada_acc = lambada_eval.evaluate(model)
        print(f"[Mode: {args.mode}] LAMBADA accuracy: {lambada_acc:.4f}")
    
if __name__ == "__main__":
    main()

