import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from model_utils import get_decoder_layers, iter_target_linears
from modules.saes_linear import SAESSVDLinear


@dataclass
class SAESConfig:
    param_ratio: float = 0.2
    damp: float = 0.01
    beta_min: float = 0.0
    beta_max: float = 0.99
    beta_cap: float = 0.95
    beta_shrink: float = 1.0
    beta_objective: str = "ratio"
    rank_align: int = 1
    include_names: tuple = ()
    exclude_names: tuple = ("lm_head",)


class SecondOrderStat:
    def __init__(self, in_features, device):
        self.h = torch.zeros((in_features, in_features), dtype=torch.float32, device=device)
        self.delta = torch.zeros_like(self.h)
        self.n = 0

    @torch.no_grad()
    def update(self, x, x_fp):
        x = _flatten_activation(x).to(device=self.h.device, dtype=torch.float32)
        x_fp = _flatten_activation(x_fp).to(device=self.h.device, dtype=torch.float32)
        if x.shape != x_fp.shape:
            raise ValueError(f"activation shape mismatch: {x.shape} vs {x_fp.shape}")
        x = x.t().contiguous()
        x_fp = x_fp.t().contiguous()
        m = x.shape[1]
        old_n = self.n
        self.n += m
        if self.n == 0:
            return
        gamma = old_n / self.n
        self.h.mul_(gamma)
        self.delta.mul_(gamma)
        scale = math.sqrt(2.0 / self.n)
        x_scaled = x.mul(scale)
        x_fp_scaled = x_fp.mul(scale)
        self.h.addmm_(x_scaled, x_scaled.t())
        self.delta.addmm_(x_fp_scaled - x_scaled, x_scaled.t())


def _flatten_activation(x):
    if isinstance(x, tuple):
        x = x[0]
    if x.dim() == 2:
        return x.reshape(-1, x.shape[-1])
    if x.dim() == 3:
        return x.reshape(-1, x.shape[-1])
    return x.flatten(0, -2)


def _module_device(module):
    for p in module.parameters(recurse=True):
        return p.device
    return torch.device("cpu")


def _model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return _module_device(model)


def _batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _forward_for_hooks(model, batch):
    body = getattr(model, "model", None)
    if body is not None:
        return body(**batch, use_cache=False)
    return model(**batch, use_cache=False)


def _rank_for_linear(linear, param_ratio, rank_align):
    rank = int(linear.weight.numel() * param_ratio / (linear.in_features + linear.out_features))
    rank = max(1, min(rank, linear.in_features, linear.out_features))
    if rank_align > 1:
        rank = int(math.ceil(rank / rank_align) * rank_align)
        rank = max(1, min(rank, linear.in_features, linear.out_features))
    return rank


def _inv_sqrt_from_cov(h, damp):
    h = torch.nan_to_num(h.float(), nan=0.0, posinf=0.0, neginf=0.0)
    h = 0.5 * (h + h.t())
    diag_mean = torch.diag(h).mean().clamp(min=1e-6)
    ridge = damp * diag_mean
    eye = torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
    for attempt in range(8):
        try:
            evals, evecs = torch.linalg.eigh(h + ridge * eye)
            floor = max(float(ridge.item()) * 1e-3, 1e-8)
            evals = evals.clamp(min=floor).rsqrt()
            inv_sqrt = evecs @ torch.diag(evals) @ evecs.t()
            inv_sqrt = torch.nan_to_num(inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
            return 0.5 * (inv_sqrt + inv_sqrt.t())
        except RuntimeError:
            ridge = ridge * 10.0
            if h.is_cuda and attempt >= 3:
                torch.cuda.empty_cache()
    evals, evecs = torch.linalg.eigh((h + ridge * eye).cpu())
    floor = max(float(ridge.cpu().item()) * 1e-3, 1e-8)
    evals = evals.clamp(min=floor).rsqrt()
    return (evecs @ torch.diag(evals) @ evecs.t()).to(h.device)


def _svd_lowrank(matrix, rank):
    matrix = torch.nan_to_num(matrix.float(), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return torch.svd_lowrank(matrix, q=rank)
    except RuntimeError:
        u, s, vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
        return u[:, :rank].to(matrix.device), s[:rank].to(matrix.device), vh[:rank, :].t().to(matrix.device)


def _frob2(x):
    return torch.sum(x.float() * x.float())


def _inner(x, y):
    return torch.sum(x.float() * y.float())


@torch.no_grad()
def aces_beta_select(weight, h, delta, rank, damp, beta_min, beta_max, beta_cap, beta_shrink, objective):
    lmat = _inv_sqrt_from_cov(h, damp)
    w = weight.float()
    s_mat = (w @ h) @ lmat
    d_mat = (w @ delta) @ lmat
    u, _, v = _svd_lowrank(s_mat, rank)

    s_proj = s_mat - u @ (u.t() @ s_mat) - (s_mat @ v) @ v.t() + u @ (u.t() @ s_mat @ v) @ v.t()
    d_proj = d_mat - u @ (u.t() @ d_mat) - (d_mat @ v) @ v.t() + u @ (u.t() @ d_mat @ v) @ v.t()

    a = _frob2(s_proj).item()
    b = _inner(s_proj, d_proj).item()
    c = _frob2(d_proj).item()
    A = _frob2(s_mat).item()
    B = _inner(s_mat, d_mat).item()
    C = _frob2(d_mat).item()

    candidates = [beta_min, beta_max]
    if objective == "energy":
        if c > 0:
            candidates.append(-b / c)
    else:
        qa = c * B - b * C
        qb = c * A - a * C
        qc = b * A - a * B
        if abs(qa) < 1e-12:
            if abs(qb) > 1e-12:
                candidates.append(-qc / qb)
        else:
            disc = qb * qb - 4.0 * qa * qc
            if disc >= 0:
                root = math.sqrt(disc)
                candidates.extend([(-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa)])

    def score(beta):
        beta = max(beta_min, min(beta, beta_max))
        beta = beta_shrink * min(beta, beta_cap)
        if objective == "energy":
            return a + 2.0 * b * beta + c * beta * beta, beta
        denom = A + 2.0 * B * beta + C * beta * beta
        value = (a + 2.0 * b * beta + c * beta * beta) / max(denom, 1e-12)
        return value, beta

    finite_scores = [score(beta) for beta in candidates if math.isfinite(beta)]
    if not finite_scores:
        return 0.0
    _, beta = min(finite_scores)
    return float(max(beta_min, min(beta, min(beta_max, beta_cap))))


@torch.no_grad()
def saes_decompose_linear(linear, stat, cfg):
    rank = _rank_for_linear(linear, cfg.param_ratio, cfg.rank_align)
    device = linear.weight.device
    h = stat.h.to(device)
    delta = stat.delta.to(device)
    lmat = _inv_sqrt_from_cov(h, cfg.damp)
    beta = aces_beta_select(
        linear.weight.data,
        h,
        delta,
        rank,
        cfg.damp,
        cfg.beta_min,
        cfg.beta_max,
        cfg.beta_cap,
        cfg.beta_shrink,
        cfg.beta_objective,
    )
    target = (linear.weight.data.float() @ (h + beta * delta)) @ lmat
    u, s, v = _svd_lowrank(target, rank)
    if not (torch.isfinite(u).all() and torch.isfinite(s).all() and torch.isfinite(v).all()):
        beta = 0.0
        target = (linear.weight.data.float() @ h) @ lmat
        u, s, v = _svd_lowrank(target, rank)
    s_sqrt = s.clamp(min=0).sqrt()
    a_weight = u.mul(s_sqrt.view(1, -1)).to(dtype=linear.weight.dtype)
    b_weight = (v.t().mul(s_sqrt.view(-1, 1)) @ lmat).to(dtype=linear.weight.dtype)
    bias = linear.bias.data.detach().clone() if linear.bias is not None else None
    return SAESSVDLinear(a_weight, b_weight, bias).to(device=device, dtype=linear.weight.dtype), beta


def _set_child(module, dotted_name, child):
    parts = dotted_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], child)


@torch.no_grad()
def collect_layer_stats(model, fp_model, layer_idx, calib_loader, cfg, fp_model_device):
    layer = get_decoder_layers(model)[layer_idx]
    fp_layer = get_decoder_layers(fp_model)[layer_idx]
    device = _module_device(layer)
    model_input_device = _model_input_device(model)
    fp_input_device = torch.device(fp_model_device) if fp_model_device != "same" else _model_input_device(fp_model)
    current_inputs = {}
    fp_inputs = {}
    stats = {}
    handles = []

    for name, linear in iter_target_linears(layer, cfg.include_names, cfg.exclude_names):
        stats[name] = SecondOrderStat(linear.in_features, device)
        handles.append(linear.register_forward_pre_hook(lambda mod, inp, key=name: current_inputs.__setitem__(key, inp[0].detach())))
    for name, linear in iter_target_linears(fp_layer, cfg.include_names, cfg.exclude_names):
        handles.append(linear.register_forward_pre_hook(lambda mod, inp, key=name: fp_inputs.__setitem__(key, inp[0].detach())))

    try:
        for batch in calib_loader:
            current_inputs.clear()
            fp_inputs.clear()
            current_out = _forward_for_hooks(model, _batch_to_device(batch, model_input_device))
            fp_out = _forward_for_hooks(fp_model, _batch_to_device(batch, fp_input_device))
            del current_out, fp_out
            for name, stat in stats.items():
                stat.update(current_inputs[name], fp_inputs[name])
    finally:
        for handle in handles:
            handle.remove()
    return stats


@torch.no_grad()
def compress_model_saes(model, fp_model, calib_loader, cfg, fp_model_device="same"):
    model.eval()
    fp_model.eval()
    layers = get_decoder_layers(model)
    beta_log = {}
    for layer_idx in tqdm(range(len(layers)), desc="SAES-SVD layers"):
        layer = layers[layer_idx]
        stats = collect_layer_stats(model, fp_model, layer_idx, calib_loader, cfg, fp_model_device)
        for name, linear in list(iter_target_linears(layer, cfg.include_names, cfg.exclude_names)):
            svd_linear, beta = saes_decompose_linear(linear, stats[name], cfg)
            _set_child(layer, name, svd_linear)
            beta_log[f"model.layers.{layer_idx}.{name}"] = beta
            if linear.weight.is_cuda:
                torch.cuda.empty_cache()
    return beta_log
