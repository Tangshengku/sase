import os
import random

import numpy as np
import torch
from datasets import load_dataset


MIXED_CALIB_NAMES = {"mix:wikitext2,evol-codealpaca,tulu-math", "wikitext2_evol-codealpaca_tulu-math", "mixture"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pad_token_id(tokenizer):
    for value in (tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id):
        if value is not None:
            return value
    raise ValueError("tokenizer must define pad_token_id, eos_token_id, or bos_token_id")


def _tokenize(tokenizer, text, seqlen, use_bos):
    if use_bos and tokenizer.bos_token:
        text = tokenizer.bos_token + text
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seqlen, padding="max_length")
    pad_id = _pad_token_id(tokenizer)
    input_ids = torch.where(enc.attention_mask.bool(), enc.input_ids, torch.full_like(enc.input_ids, pad_id))
    return {"input_ids": input_ids, "attention_mask": enc.attention_mask}


def _sample_text_corpus(text, tokenizer, nsamples, seqlen, use_bos):
    samples = []
    for _ in range(nsamples):
        start = random.randint(0, max(0, len(text) - seqlen * 8 - 1))
        chunk = text[start : start + seqlen * 8]
        first_period = chunk.find(".")
        if first_period >= 0:
            chunk = chunk[first_period + 1 :]
        samples.append(_tokenize(tokenizer, chunk.strip(), seqlen, use_bos))
    return samples


def _format_messages(messages):
    turns = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if content:
            turns.append(f"{role}: {content}")
    return "\n".join(turns)


def _format_evol_codealpaca(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    parts = [f"Instruction:\n{instruction}"]
    if input_text:
        parts.append(f"Input:\n{input_text}")
    if output:
        parts.append(f"Response:\n{output}")
    return "\n\n".join(parts)


def _format_tulu_math(example):
    if "messages" in example:
        return _format_messages(example["messages"])
    if "prompt" in example:
        return example["prompt"]
    return example.get("text", "")


def _sample_instruction_dataset(dataset, formatter, tokenizer, seqlen, use_bos):
    texts = []
    token_count = 0
    attempts = 0
    while token_count < seqlen and attempts < 64:
        example = dataset[random.randint(0, len(dataset) - 1)]
        text = formatter(example).strip()
        attempts += 1
        if not text:
            continue
        texts.append(text)
        token_count += len(tokenizer.encode(text, add_special_tokens=False))
    return _tokenize(tokenizer, "\n\n".join(texts), seqlen, use_bos)


def _get_mixed_calib_data(tokenizer, nsamples, seqlen, use_bos):
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wikitext = "\n\n".join(wikitext["text"])
    evol_codealpaca = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
    tulu_math = load_dataset("allenai/tulu-3-sft-personas-math", split="train")

    sources = [
        lambda: _sample_text_corpus(wikitext, tokenizer, 1, seqlen, use_bos)[0],
        lambda: _sample_instruction_dataset(evol_codealpaca, _format_evol_codealpaca, tokenizer, seqlen, use_bos),
        lambda: _sample_instruction_dataset(tulu_math, _format_tulu_math, tokenizer, seqlen, use_bos),
    ]

    samples = [sources[i % len(sources)]() for i in range(nsamples)]
    random.shuffle(samples)
    return samples


def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=0, use_bos=False, cache_dir="cache"):
    set_seed(seed)
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = name.replace("/", "_").replace(":", "_").replace(",", "_")
    cache_file = os.path.join(
        cache_dir,
        f"{safe_name}_{model_id.replace('/', '_')}_{nsamples}_{seqlen}_{seed}_bos{use_bos}.pt",
    )
    if os.path.exists(cache_file):
        return torch.load(cache_file, map_location="cpu")

    if name in MIXED_CALIB_NAMES:
        samples = _get_mixed_calib_data(tokenizer, nsamples, seqlen, use_bos)
        torch.save(samples, cache_file)
        return samples
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(dataset["text"])
    elif name == "c4":
        dataset = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        text = "\n\n".join(dataset["text"])
    else:
        raise NotImplementedError(f"Unsupported calibration dataset: {name}")

    samples = _sample_text_corpus(text, tokenizer, nsamples, seqlen, use_bos)
    torch.save(samples, cache_file)
    return samples
