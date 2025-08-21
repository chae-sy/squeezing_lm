import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial

# -----------------------
# Your eval_ppl (verbatim)
# -----------------------
import math
import torch

@torch.no_grad()
def eval_ppl(model, tokenizer, dataset, batch_size=8, max_batches=None, device="cuda"):
    model.eval().to(device)
    total_nll = 0.0
    total_tokens = 0

    loader = torch.utils.data.DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False
    )

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)      # [B, T], no padding in our block dataset
        labels = input_ids.clone()

        # Forward; HF loss is mean over tokens != -100
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Tokens counted this batch
        if (labels == -100).any():
            batch_tokens = (labels != -100).sum().item()
        else:
            batch_tokens = labels.numel()

        # Safety: skip bad batches
        if not torch.isfinite(loss):
            continue

        total_nll += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens

    ppl = math.exp(total_nll / max(1, total_tokens))
    return ppl

# -----------------------
# Data prep
# -----------------------
def build_wikitext2(tokenizer, block_size=1024):
    """
    Returns a DatasetDict with 'validation' split containing fixed-length chunks
    as {'input_ids': LongTensor}.
    """
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(example):
        return tokenizer(example["text"])

    tokenized = raw.map(tokenize, batched=True, num_proc=1, remove_columns=raw["train"].column_names)

    def group_texts(examples):
        # Concatenate
        concatenated = []
        for input_ids in examples["input_ids"]:
            concatenated.extend(input_ids)
        total_len = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_len]
        # Split into blocks
        result = {"input_ids": [torch.tensor(concatenated[i:i+block_size], dtype=torch.long)
                                for i in range(0, total_len, block_size)]}
        return result

    # We only need validation for ppl
    val_grouped = tokenized["validation"].map(
        group_texts,
        batched=True,
        remove_columns=tokenized["validation"].column_names
    )

    # Make sure items are dicts with tensors (PyTorch default collate will stack them)
    val_grouped.set_format(type="torch", columns=["input_ids"])

    return {"validation": val_grouped}

# -----------------------
# Simple W8A8 fake-quant PTQ
# -----------------------
def quantize_per_channel_int8(w: torch.Tensor, axis: int = 0, eps: float = 1e-8):
    """
    Symmetric per-channel int8 quantization for weights.
    Returns (w_int8, scale) where:
      - w_int8: int8 tensor same shape as w
      - scale: float32 scale per 'axis' channel (shape matches the axis length)
    """
    # max abs per channel
    max_abs = w.abs().amax(dim=axis, keepdim=True).clamp(min=eps)
    scale = (max_abs / 127.0).squeeze(axis)  # shape: out_channels if axis==0
    # reshape for broadcast when quantizing
    scale_b = scale.unsqueeze(axis)
    w_int = torch.round((w / scale_b).clamp(-128, 127)).to(torch.int8)
    return w_int, scale

class ActFakeQuant(nn.Module):
    """
    Per-tensor symmetric int8 fake quantization for activations.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            max_abs = x.abs().amax().clamp(min=self.eps)
            a_scale = max_abs / 127.0
            x_q = torch.round((x / a_scale).clamp(-128, 127))
            x_dq = x_q * a_scale
        return x_dq

class QuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear doing:
      - int8 per-channel weight quant (precomputed)
      - per-tensor activation fake-quant (runtime)
      - dequantized matmul in float (int8 * scale pre-dequantized once)
    This simulates W8A8 effects while staying simple & robust.
    """
    def __init__(self, lin: nn.Linear):
        super().__init__()
        assert isinstance(lin, nn.Linear)
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.bias = None
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.detach().clone())

        # Quantize weights per OUT channel (axis=0 on [out, in])
        w = lin.weight.detach().to(torch.float32).cpu()
        w_int8, w_scale = quantize_per_channel_int8(w, axis=1)
        self.register_buffer("w_int8", w_int8, persistent=False)
        self.register_buffer("w_scale", w_scale, persistent=False)

        # Precompute dequantized weight in float32 to avoid per-forward overhead
        # (each row scaled by corresponding w_scale)
        w_deq = (self.w_int8.float() * self.w_scale.unsqueeze(1))
        self.register_buffer("w_deq", w_deq, persistent=False)

        self.act_q = ActFakeQuant()

    def forward(self, x: torch.Tensor):
        x_q = self.act_q(x)  # fake-quant activations
        # x @ W^T + b ; W_deq is [out, in]
        y = torch.matmul(x_q, self.w_deq.t())
        if self.bias is not None:
            y = y + self.bias
        return y

def replace_linear_with_quant(module: nn.Module):
    """
    Recursively replace nn.Linear with QuantLinear.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantLinear(child))
        else:
            replace_linear_with_quant(child)

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fp16", "w8a8"], required=True)
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        # For causal LM, set pad to eos to keep batching simple
        tokenizer.pad_token = tokenizer.eos_token

    print("Building WikiText-2 (raw) validation set...")
    dataset = build_wikitext2(tokenizer, block_size=args.block_size)

    print(f"Loading model: {args.model_name}")
    # Keep in float32 first (safer for weight processing), downcast later if fp16
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if args.mode == "fp16":
        print("Converting model to FP16…")
        model = model.half()
    elif args.mode == "w8a8":
        print("Applying W8A8 (fake-quant) PTQ to Linear layers…")
        model = model.float()  # ensure base is float32 before quant
        replace_linear_with_quant(model)
    else:
        raise ValueError("Unknown mode")

    # Ensure model uses correct pad token id for loss masking
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Evaluating perplexity…")
    ppl = eval_ppl(model, tokenizer, dataset, batch_size=args.eval_bs, max_batches=args.max_batches, device=args.device)
    print(f"[Mode: {args.mode}] Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()

