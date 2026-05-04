# SAES-SVD

This directory contains a practical implementation of the SAES-SVD method from
`2602.03051v1.pdf`.

The compressor targets Hugging Face causal LMs with LLaMA-style decoder layers,
including Qwen3-8B and Mistral-7B. It uses:

- streaming second-order statistics `H = X X^T`
- cumulative error statistics `Delta = (X_fp - X) X^T`
- ACES adaptive beta selection
- closed-form SAES-SVD factorization

## Install

```bash
pip install -r requirements.txt
```

## Qwen3-8B, 0.2 parameter ratio

```bash
CUDA_VISIBLE_DEVICES=0 python saes.py \
  --model_id Qwen/Qwen3-8B \
  --param_ratio_target 0.2 \
  --n_calib_samples 128 \
  --calib_batch_size 4 \
  --seqlen 2048 \
  --calib_dataset mix:wikitext2,evol-codealpaca,tulu-math \
  --eval_c4_ppl \
  --save_model output/Qwen3-8B-SAES-0.2 \
  --trust_remote_code
```

## Mistral-7B, 0.2 parameter ratio

```bash
CUDA_VISIBLE_DEVICES=0 python saes.py \
  --model_id mistralai/Mistral-7B-v0.1 \
  --param_ratio_target 0.2 \
  --n_calib_samples 128 \
  --calib_batch_size 4 \
  --seqlen 2048 \
  --calib_dataset mix:wikitext2,evol-codealpaca,tulu-math \
  --eval_c4_ppl \
  --save_model output/Mistral-7B-SAES-0.2 \
  --trust_remote_code
```

The saved model is exported in dense Hugging Face style: all SAES low-rank
factors are multiplied back into ordinary `nn.Linear` weights before saving, so
`lm-eval-harness` can load it as a normal model directory.

`--eval_c4_ppl` computes a quick C4 validation perplexity on the compressed
model and writes `eval_c4_ppl.json` into the output directory. Tune runtime with
`--eval_c4_samples`, `--eval_c4_seqlen`, and `--eval_c4_batch_size`.

For limited memory, lower `--n_calib_samples` and `--seqlen`, or add
`--fp_model_device cpu` to keep the full-precision reference model on CPU.
# sase
