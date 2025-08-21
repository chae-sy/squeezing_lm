import math, argparse
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
        batch_tokens = (labels != -100).sum().item() if (labels == -100).any() else labels.numel()
        if not torch.isfinite(loss): continue
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
# Quant utils
# -----------------------
def quantize_per_channel_int8(w: torch.Tensor, axis: int, eps: float = 1e-8):
    max_abs = w.abs().amax(dim=axis, keepdim=True).clamp(min=eps)
    scale = (max_abs / 127.0).squeeze(axis).to(torch.float32)     # [w.shape[axis]]
    w_int8 = torch.round((w / max_abs).clamp(-1, 1) * 127).to(torch.int8)
    return w_int8, scale

class ActFakeQuant(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            max_abs = x.abs().amax().clamp(min=self.eps)
            a_scale = max_abs / 127.0
            x_q = torch.round((x / a_scale).clamp(-128, 127))
            return x_q * a_scale

class QuantLinear(nn.Module):
    """Plain W8A8 fake-quant linear"""
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None
        w = lin.weight.detach().to(torch.float32).cpu()                 # [out, in]
        w_int8, w_scale = quantize_per_channel_int8(w, axis=1)          # per-output-channel quant
        self.register_buffer("w_int8", w_int8, persistent=False)
        self.register_buffer("w_scale", w_scale, persistent=False)
        self.register_buffer("w_deq", (self.w_int8.float() * self.w_scale.unsqueeze(1)), persistent=False)
        self.act_q = ActFakeQuant()
    def forward(self, x: torch.Tensor):
        x_q = self.act_q(x)
        y = torch.matmul(x_q, self.w_deq.t())
        if self.bias is not None: y = y + self.bias
        return y

def replace_linear_with_quant(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear(child))
        else:
            replace_linear_with_quant(child)

# -----------------------
# SmoothQuant
# -----------------------
class SQQuantLinear(nn.Module):
    """
    SmoothQuant + W8A8 fake quant in one module.
    Applies input scaling (x * s) and rescales weights column-wise by 1/s, then quantizes.
    """
    def __init__(self, lin: nn.Linear, s_vec: torch.Tensor):
        super().__init__()
        assert isinstance(lin, nn.Linear)
        assert s_vec.ndim == 1 and s_vec.shape[0] == lin.in_features
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.register_buffer("s_vec", s_vec.to(torch.float32), persistent=False)  # [in]

        self.bias = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None

        # Column-wise rescale: W[:, j] *= (1 / s_j)
        w = lin.weight.detach().to(torch.float32).cpu()                              # [out, in]
        w = w * ( self.s_vec.unsqueeze(0))                                      # broadcast over rows

        # Quantize per-output-channel (reduce over input dim)
        w_int8, w_scale = quantize_per_channel_int8(w, axis=1)
        self.register_buffer("w_int8", w_int8, persistent=False)
        self.register_buffer("w_scale", w_scale, persistent=False)
        self.register_buffer("w_deq", (self.w_int8.float() * self.w_scale.unsqueeze(1)), persistent=False)

        self.act_q = ActFakeQuant()

    def forward(self, x: torch.Tensor):
        x = x / self.s_vec          # activation scaling
        x_q = self.act_q(x)         # fake-quant activation
        y = torch.matmul(x_q, self.w_deq.t())
        if self.bias is not None: y = y + self.bias
        return y

@torch.no_grad()
def collect_activation_amax_per_linear(model: nn.Module, calib_loader, device, max_batches=32):
    """
    Returns dict: module -> amax_per_input_channel (shape [in_features])
    """
    model.eval().to(device)
    lin_modules = []
    amax_dict = {}

    # discover all Linear modules
    for m in model.modules():
        if isinstance(m, nn.Linear):
            lin_modules.append(m)
            amax_dict[m] = torch.zeros(m.in_features, dtype=torch.float32, device=device)

    hooks = []
    def make_hook(m):
        def hook(module, inputs):
            x = inputs[0]  # [B, T, in]
            # reduce over batch/time dims, keep per-input-channel abs max
            cur = x.detach().abs().amax(dim=tuple(range(x.ndim-1)))  # last dim is in_features
            amax_dict[module].copy_(torch.maximum(amax_dict[module], cur))
        return hook

    for m in lin_modules:
        hooks.append(m.register_forward_pre_hook(make_hook(m), with_kwargs=False))

    # run a few calibration batches
    for i, batch in enumerate(calib_loader):
        if i >= max_batches: break
        input_ids = batch["input_ids"].to(device)
        _ = model(input_ids=input_ids)

    # cleanup hooks
    for h in hooks:
        h.remove()

    # move to cpu for later processing
    for k in amax_dict:
        amax_dict[k] = amax_dict[k].detach().cpu()
    return amax_dict

def apply_smoothquant_and_quantize(model: nn.Module, act_amax: dict, alpha: float = 0.5, eps: float = 1e-8):
    def transform(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                # --- Add this line to skip lm_head and similar layers ---
                if any(k in name.lower() for k in ["lm_head"]):
                    continue

                a = act_amax.get(child, None)
                if a is None:
                    s = torch.ones(child.in_features, dtype=torch.float32)
                else:
                    w_col_max = child.weight.detach().abs().amax(dim=0).cpu()
                    a = torch.clamp(a, min=eps)
                    w_col_max = torch.clamp(w_col_max, min=eps)
                    s = (a.pow(alpha)) / (w_col_max.pow(1.0 - alpha))
                    s = torch.clamp(s, min=1e-3, max=1e3)
                    s = s/s.median()
                setattr(module, name, SQQuantLinear(child, s))
            else:
                transform(child)
    transform(model)


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp16", "w8a8", "w8a8_sq"], required=True)
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # SmoothQuant args
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant α in [0,1]")
    parser.add_argument("--calib_batches", type=int, default=32, help="Num batches for activation calibration")
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
        print("Applying W8A8 (fake-quant) PTQ to Linear layers…")
        model = model.float()
        replace_linear_with_quant(model)

    elif args.mode == "w8a8_sq":
        print(f"Calibrating activations for SmoothQuant (alpha={args.alpha})…")
        model = model.float()
        calib_loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=args.eval_bs, shuffle=False)
        act_amax = collect_activation_amax_per_linear(model, calib_loader, device=args.device, max_batches=args.calib_batches)
        print("Applying SmoothQuant + W8A8…")
        apply_smoothquant_and_quantize(model, act_amax, alpha=args.alpha)

    else:
        raise ValueError("Unknown mode")

    model.config.pad_token_id = tokenizer.pad_token_id

    print("Evaluating perplexity…")
    ppl = eval_ppl(model, tokenizer, dataset, batch_size=args.eval_bs,
                   max_batches=args.max_batches, device=args.device)
    print(f"[Mode: {args.mode}] Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()

