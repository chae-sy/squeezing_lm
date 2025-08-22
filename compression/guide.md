## Set up
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate safetensors
```
## 1️⃣ Perplexity Evaluation for quantized models

Script: opt_quant_ppl_eval.py

Supported Modes

fp16 — Half-precision baseline

w8a8 — Naive W8A8 quantization

sq — SmoothQuant with configurable W/A bitwidths

lora_qat_w4a4 — LoRA-QAT with W4A4 adapters

lora_qat_w4a4_sq — SmoothQuant baseline (W8A8) with LoRA-QAT adapters (A4)

Examples
```
# FP16 PPL
python opt_quant_ppl_eval.py --mode fp16

# SmoothQuant W8A8
python opt_quant_ppl_eval.py --mode sq --w_bits 8 --a_bits 8 --alpha 0.55 --calib_batches 64 --quantile 0.999

# SmoothQuant W8A8
python opt_quant_ppl_eval.py --mode sq --w_bits 6 --a_bits 8 --alpha 0.55 --calib_batches 64 --quantile 0.999

# SmoothQuant W8A8
python opt_quant_ppl_eval.py --mode sq --w_bits 6 --a_bits 6 --alpha 0.55 --calib_batches 64 --quantile 0.999

# SmoothQuant W8A8
python opt_quant_ppl_eval.py --mode sq --w_bits 6 --a_bits 6 --alpha 0.6 --calib_batches 64 --quantile 0.9992

# SmoothQuant W8A8
python opt_quant_ppl_eval.py --mode sq --w_bits 4 --a_bits 8 --alpha 0.75 --calib_batches 64 --quantile 0.9995

# SmoothQuant W8A8
python opt_quant_ppl_eval.py --mode sq --w_bits 4 --a_bits 4 --alpha 0.75 --calib_batches 64 --quantile 0.9995

# LoRA-QAT with SmoothQuant baseline, plus LAMBADA
python opt_quant_ppl_eval.py --mode lora_qat_w4a4_sq --eval_lambada --lambada_samples 1000 \
  --qat_steps 800 --train_bs 8 --eval_bs 8 --alpha 0.80 --quantile 0.999
``` 

## 2️⃣ LAMBADA Accuracy Evaluation for quantized models 

Script: opt_quant_acc_eval.py

```
# W8A8 accuracy
python opt_quant_acc_eval.py --mode w8a8 --lambada_split validation[:1000]

# SmoothQuant W6A8 accuracy
python opt_quant_acc_eval.py --mode sq --w_bits 6 --a_bits 8 --alpha 0.55 --calib_batches 64

# W4a4 with lora QAT and SmoothQuant with 10 qat steps
python opt_quant_acc_eval.py --mode lora_qat_w4a4_sq --model_name facebook/opt-125m \
--block_size 1024 --train_bs 8 --eval_bs 8 --calib_batches 32 --quantile 0.999 \
--alpha 0.80 --qat_steps 10 --qat_lr 3e-4 --eval_lambada --lambada_samples 1000
```

## 3️⃣ Perplexity Evaluation for GQA Conversion
Conversion and Uptraining

Script: opt_gqa_ppl_eval.py

```
# Convert to GQA (G=6), evaluate PPL
python opt_gqa_ppl_eval.py --groups 6 --save_dir ./opt125m-gqa6-up --convert_only --eval_ppl

# Convert to GQA (G=3), uptrain briefly, then test speed
python opt_gqa_ppl_eval.py --groups 3 --train --max_steps 1000 --save_dir ./opt125m-gqa3-up --eval_speed
```

## 4️⃣ Accuracy Evaluation for GQA Conversion

Script: opt_gqa_acc_eval.py

```
# Evaluate accuracy on GQA3-up model
python opt_gqa_acc_eval.py --load_dir ./opt125m-gqa3-up --max_samples 1000

# Evaluate accuracy on GQA6-up model with specific GPU
python opt_gqa_acc_eval.py --load_dir ./opt125m-gqa6-up --max_samples 1000 --cuda_visible_devices 0
```

## 5️⃣ LoRA-QAT with MQA 

Script: opt_lora_qat_sq_gqa.py

```
# LoRA-QAT (W4A4 adapters) on top of uptrained GQA3 checkpoint
python opt_lora_qat_sq_gqa.py --mode lora_qat_w4a4_sq \
  --up_dir ./opt125m-gqa3-up --eval_lambada --lambada_samples 1000 \
  --alpha 0.80 --calib_batches 64 --qat_steps 100 --qat_lr 3e-4 --train_bs 8 --eval_bs 8
```
