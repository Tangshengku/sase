import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from datautils import get_calib_data
from saes_svd import SAESConfig, compress_model_saes
from save_utils import save_dense_hf


def make_calib_batches(samples, batch_size):
    if batch_size <= 1:
        return samples
    batches = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        keys = chunk[0].keys()
        batches.append({key: torch.cat([sample[key] for sample in chunk], dim=0) for key in keys})
    return batches


def _model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        for param in model.parameters():
            return param.device
    return torch.device("cpu")


@torch.no_grad()
def evaluate_c4_ppl(model, tokenizer, seqlen=2048, nsamples=256, batch_size=1):
    model.eval()
    dataset = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    target_tokens = nsamples * seqlen + 1
    token_ids = []
    for example in dataset:
        text = example.get("text", "")
        if not text:
            continue
        token_ids.extend(tokenizer.encode(text + "\n\n", add_special_tokens=False))
        if len(token_ids) >= target_tokens:
            break
    input_ids = torch.tensor(token_ids[:target_tokens], dtype=torch.long).unsqueeze(0)
    max_chunks = max(0, (input_ids.numel() - 1) // seqlen)
    nsamples = min(nsamples, max_chunks)
    if nsamples <= 0:
        raise ValueError("not enough C4 validation tokens for perplexity evaluation")

    device = _model_input_device(model)
    nlls = []
    for start in tqdm(range(0, nsamples, batch_size), desc="C4 ppl"):
        chunks = []
        for idx in range(start, min(start + batch_size, nsamples)):
            begin = idx * seqlen
            chunks.append(input_ids[:, begin : begin + seqlen])
        batch = torch.cat(chunks, dim=0).to(device)
        outputs = model(input_ids=batch, labels=batch, use_cache=False)
        nlls.append(outputs.loss.detach().float() * (batch.size(1) - 1) * batch.size(0))

    total_nll = torch.stack(nlls).sum()
    total_tokens = nsamples * (seqlen - 1)
    ppl = torch.exp(total_nll / total_tokens)
    return float(ppl.item())


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    fp_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.fp_model_device == "cpu":
        fp_kwargs["device_map"] = {"": "cpu"}
    else:
        fp_kwargs["device_map"] = args.device_map
    fp_model = AutoModelForCausalLM.from_pretrained(args.model_id, **fp_kwargs)

    calib_loader = get_calib_data(
        args.calib_dataset,
        tokenizer,
        args.model_id,
        args.n_calib_samples,
        seqlen=args.seqlen,
        seed=args.seed,
        use_bos=args.use_bos,
    )
    calib_loader = make_calib_batches(calib_loader, args.calib_batch_size)

    if args.compression_ratio is not None:
        if args.param_ratio_target is not None:
            raise ValueError("Use either --compression_ratio or --param_ratio_target, not both")
        param_ratio = 1.0 - args.compression_ratio
    else:
        param_ratio = args.param_ratio_target
    if not 0.0 < param_ratio <= 1.0:
        raise ValueError(f"retained parameter ratio must be in (0, 1], got {param_ratio}")
    print(f"retained_param_ratio={param_ratio:.4f}")

    cfg = SAESConfig(
        param_ratio=param_ratio,
        damp=args.damp,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_cap=args.beta_cap,
        beta_shrink=args.beta_shrink,
        fixed_beta=args.fixed_beta,
        beta_objective=args.beta_objective,
        rank_align=args.rank_align,
        svd_oversample=args.svd_oversample,
        svd_niter=args.svd_niter,
        svd_method=args.svd_method,
        decomposition=args.decomposition,
        include_names=tuple(args.include_names.split(",")) if args.include_names else (),
        exclude_names=tuple(args.exclude_names.split(",")) if args.exclude_names else (),
    )
    beta_log = compress_model_saes(model, fp_model, calib_loader, cfg, fp_model_device=args.fp_model_device)
    if beta_log:
        beta_values = torch.tensor(list(beta_log.values()), dtype=torch.float32)
        print(
            "beta_summary="
            f"min:{beta_values.min().item():.4f},"
            f"mean:{beta_values.mean().item():.4f},"
            f"max:{beta_values.max().item():.4f}"
        )

    c4_ppl = None
    if args.eval_c4_ppl:
        c4_ppl = evaluate_c4_ppl(
            model,
            tokenizer,
            seqlen=args.eval_c4_seqlen,
            nsamples=args.eval_c4_samples,
            batch_size=args.eval_c4_batch_size,
        )
        print(f"c4_ppl={c4_ppl:.4f}")

    if args.save_model:
        save_dense_hf(model, tokenizer, args.save_model)
        with open(os.path.join(args.save_model, "saes_betas.json"), "w") as f:
            json.dump(beta_log, f, indent=2, sort_keys=True)
        if c4_ppl is not None:
            with open(os.path.join(args.save_model, "eval_c4_ppl.json"), "w") as f:
                json.dump(
                    {
                        "c4_ppl": c4_ppl,
                        "eval_c4_samples": args.eval_c4_samples,
                        "eval_c4_seqlen": args.eval_c4_seqlen,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=0.8,
        help="fraction of Linear parameters to retain after low-rank factorization; 0.2 keeps 20%",
    )
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=None,
        help="fraction of Linear parameters to remove; 0.2 keeps about 80%",
    )
    parser.add_argument("--n_calib_samples", type=int, default=128)
    parser.add_argument("--calib_batch_size", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="mix:wikitext2,evol-codealpaca,tulu-math",
        choices=["wikitext2", "c4", "mix:wikitext2,evol-codealpaca,tulu-math", "wikitext2_evol-codealpaca_tulu-math", "mixture"],
    )
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--use_bos", action="store_true")
    parser.add_argument("--save_model", type=str, default="")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--fp_model_device", type=str, default="same", choices=["same", "cpu"])
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--beta_min", type=float, default=0.0)
    parser.add_argument("--beta_max", type=float, default=0.99)
    parser.add_argument("--beta_cap", type=float, default=0.5)
    parser.add_argument("--beta_shrink", type=float, default=1.0)
    parser.add_argument("--fixed_beta", type=float, default=None, help="disable ACES and use a fixed beta, e.g. 0.0")
    parser.add_argument("--beta_objective", type=str, default="ratio", choices=["ratio", "energy"])
    parser.add_argument("--rank_align", type=int, default=1)
    parser.add_argument("--svd_oversample", type=int, default=32)
    parser.add_argument("--svd_niter", type=int, default=4)
    parser.add_argument("--svd_method", type=str, default="exact", choices=["exact", "randomized"])
    parser.add_argument("--decomposition", type=str, default="saes", choices=["vanilla", "asvd", "saes"])
    parser.add_argument("--eval_c4_ppl", action="store_true", help="compute quick C4 validation perplexity after compression")
    parser.add_argument("--eval_c4_samples", type=int, default=256)
    parser.add_argument("--eval_c4_seqlen", type=int, default=2048)
    parser.add_argument("--eval_c4_batch_size", type=int, default=1)
    parser.add_argument(
        "--include_names",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="comma-separated Linear leaf names to compress",
    )
    parser.add_argument("--exclude_names", type=str, default="lm_head")
    main(parser.parse_args())
