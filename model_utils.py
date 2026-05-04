import torch.nn as nn


def get_model_type(model):
    return getattr(model.config, "model_type", "").lower()


def get_model_name(model):
    return getattr(model.config, "_name_or_path", "").lower()


def get_decoder_layers(model):
    model_type = get_model_type(model)
    model_name = get_model_name(model)
    if model_type in {"llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2"}:
        return model.model.layers
    if any(name in model_name for name in ("llama", "mistral", "qwen", "gemma")):
        return model.model.layers
    raise NotImplementedError(
        f"Unsupported decoder layout: model_type={model_type!r}, name={model_name!r}"
    )


def iter_target_linears(module, include_names=None, exclude_names=None):
    include_names = set(include_names or [])
    exclude_names = set(exclude_names or [])
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if include_names and leaf not in include_names and name not in include_names:
            continue
        if leaf in exclude_names or name in exclude_names:
            continue
        yield name, child

