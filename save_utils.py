import os

import torch
import torch.nn as nn

from modules.saes_linear import SAESSVDLinear


SAES_MODELING_CODE = r'''import torch
import torch.nn as nn

try:
    from transformers import MistralForCausalLM
except ImportError:
    try:
        from transformers.models.mistral.modeling_mistral import MistralForCausalLM
    except ImportError:
        MistralForCausalLM = None

try:
    from transformers import Qwen3ForCausalLM
except ImportError:
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    except ImportError:
        try:
            from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM
        except ImportError:
            try:
                from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as Qwen3ForCausalLM
            except ImportError:
                Qwen3ForCausalLM = None


class SAESSVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.truncation_rank = rank

    def forward(self, x):
        return self.ALinear(self.BLinear(x))


def _set_module(root, name, module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _replace_saes_linears(model, config):
    for name, info in getattr(config, "saes_linear_info", {}).items():
        _set_module(
            model,
            name,
            SAESSVDLinear(info["in_features"], info["out_features"], info["rank"], bias=info["bias"]),
        )


if MistralForCausalLM is not None:
    class SAESMistralForCausalLM(MistralForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            _replace_saes_linears(self, config)
else:
    class SAESMistralForCausalLM(nn.Module):
        def __init__(self, config):
            raise ImportError("MistralForCausalLM is unavailable in this transformers installation.")


if Qwen3ForCausalLM is not None:
    class SAESQwen3ForCausalLM(Qwen3ForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            _replace_saes_linears(self, config)
else:
    class SAESQwen3ForCausalLM(nn.Module):
        def __init__(self, config):
            raise ImportError("Qwen3ForCausalLM/Qwen2ForCausalLM is unavailable in this transformers installation.")
'''


def _auto_class_name(model):
    model_type = getattr(model.config, "model_type", "").lower()
    model_name = getattr(model.config, "_name_or_path", "").lower()
    if model_type == "mistral" or "mistral" in model_name:
        return "SAESMistralForCausalLM"
    if model_type in {"qwen3", "qwen2"} or "qwen" in model_name:
        return "SAESQwen3ForCausalLM"
    raise NotImplementedError(f"HF export supports Mistral and Qwen. Got model_type={model_type!r}.")


def _collect_linear_info(model):
    info = {}
    for name, module in model.named_modules():
        if isinstance(module, SAESSVDLinear):
            info[name] = {
                "in_features": module.BLinear.in_features,
                "out_features": module.ALinear.out_features,
                "rank": module.truncation_rank,
                "bias": module.ALinear.bias is not None,
            }
    return info


def save_saes_hf(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.config.saes_linear_info = _collect_linear_info(model)
    model.config.architectures = [_auto_class_name(model)]
    model.config.auto_map = {"AutoModelForCausalLM": f"modeling_saes.{model.config.architectures[0]}"}
    with open(os.path.join(output_dir, "modeling_saes.py"), "w") as f:
        f.write(SAES_MODELING_CODE)
    tokenizer.save_pretrained(output_dir)
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(output_dir)
    print(f"Saved HF-style SAES-SVD model to {output_dir}")


def _set_module(root, name, module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


@torch.no_grad()
def densify_saes_linears(model):
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, SAESSVDLinear):
            a_linear = module.ALinear
            b_linear = module.BLinear
            dense = nn.Linear(
                b_linear.in_features,
                a_linear.out_features,
                bias=a_linear.bias is not None,
                device=a_linear.weight.device,
                dtype=a_linear.weight.dtype,
            )
            dense.weight.data.copy_((a_linear.weight.float() @ b_linear.weight.float()).to(a_linear.weight.dtype))
            if a_linear.bias is not None:
                dense.bias.data.copy_(a_linear.bias.data)
            replacements.append((name, dense))
    for name, dense in replacements:
        _set_module(model, name, dense)
    return len(replacements)


def save_dense_hf(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    replaced = densify_saes_linears(model)
    for attr in ("saes_linear_info", "auto_map"):
        if hasattr(model.config, attr):
            delattr(model.config, attr)
    tokenizer.save_pretrained(output_dir)
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(output_dir)
    print(f"Saved dense HF model to {output_dir} after materializing {replaced} SAES-SVD layers")
