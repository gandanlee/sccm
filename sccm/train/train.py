from tqdm import tqdm
from sccm.utils.utils import to_cuda
import sccm
import torch
import wandb

def log_param_statistics(named_parameters, norm_type = 2):
    named_parameters = list(named_parameters)
    grads = [p.grad for n, p in named_parameters if p.grad is not None]
    weight_norms = [p.norm(p=norm_type) for n, p in named_parameters if p.grad is not None]
    names = [n for n,p in named_parameters if p.grad is not None]
    param_norm = torch.stack(weight_norms).norm(p=norm_type)
    device = grads[0].device
    grad_norms = torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads])
    nans_or_infs = torch.isinf(grad_norms) | torch.isnan(grad_norms)
    nan_inf_names = [name for name, naninf in zip(names, nans_or_infs) if naninf]
    total_grad_norm = torch.norm(grad_norms, norm_type)
    if torch.any(nans_or_infs):
        print(f"These params have nan or inf grads: {nan_inf_names}")
    wandb.log({"grad_norm": total_grad_norm.item()}, step = sccm.GLOBAL_STEP)
    wandb.log({"param_norm": param_norm.item()}, step = sccm.GLOBAL_STEP)

def _apply_grad_clip(named_parameters, default_clip, by_substring):
    """Apply per-substring grad clipping; falls back to a global clip.

    When ``by_substring`` is None or empty, performs a single global
    ``clip_grad_norm_(all_params, default_clip)`` (existing behaviour).

    When provided as a dict ``{substring: clip_value}``, parameters matching
    the first substring (in dict-iteration order) are clipped per their own
    sub-group; remaining parameters fall back to ``default_clip``. Each group
    has its OWN total-norm budget — this is the key fix for sparse-trainable
    refiner experiments where a global clip 0.01 throttles ~50K refiner
    params even though they have natural grad_norm ~0.55.
    """
    if not by_substring:
        torch.nn.utils.clip_grad_norm_([p for _, p in named_parameters if p.grad is not None],
                                       default_clip)
        return

    grouped = {sub: [] for sub in by_substring.keys()}
    fallback = []
    for name, p in named_parameters:
        if p.grad is None:
            continue
        for sub in by_substring.keys():
            if sub in name:
                grouped[sub].append(p)
                break
        else:
            fallback.append(p)
    for sub, params in grouped.items():
        if params:
            torch.nn.utils.clip_grad_norm_(params, by_substring[sub])
    if fallback:
        torch.nn.utils.clip_grad_norm_(fallback, default_clip)


def _log_gacv2_diagnostics(model):
    """Walk modules to find SphereCovisMatcher and log its GAC-v2 stats.
    Only logs if the matcher has gacv2_enabled=True and exposed _last_gacv2_* attrs."""
    try:
        for m in model.modules():
            if getattr(m, 'gacv2_enabled', False) and hasattr(m, '_last_gacv2_gamma'):
                wandb.log({
                    'gacv2/gamma_eff':     m._last_gacv2_gamma,
                    'gacv2/gamma_raw':     m._last_gacv2_gamma_raw,
                    'gacv2/d_geo_mean':    m._last_gacv2_d_geo_mean,
                    'gacv2/d_geo_max':     m._last_gacv2_d_geo_max,
                    'gacv2/warmup_factor': m._last_gacv2_warmup_factor,
                }, step=sccm.GLOBAL_STEP)
                return
    except Exception:
        pass


def _log_alac_diagnostics(model):
    """Walk modules to find a SphereCovisHead with alac_enabled and log α(x) stats."""
    try:
        for m in model.modules():
            if getattr(m, 'alac_enabled', False) and hasattr(m, '_last_alac_alpha_mean'):
                wandb.log({
                    'alac/alpha_mean': m._last_alac_alpha_mean,
                    'alac/alpha_std':  m._last_alac_alpha_std,
                    'alac/alpha_min':  m._last_alac_alpha_min,
                    'alac/alpha_max':  m._last_alac_alpha_max,
                }, step=sccm.GLOBAL_STEP)
                return
    except Exception:
        pass


def _log_bsc_diagnostics(model):
    """Walk modules to find SphereCovisMatcher with bsc_enabled and log β + Δlogit stats."""
    try:
        for m in model.modules():
            if getattr(m, 'bsc_enabled', False) and hasattr(m, '_last_bsc_beta'):
                wandb.log({
                    'bsc/beta':              m._last_bsc_beta,
                    'bsc/logE_mu_B_mean':    m._last_bsc_logE_mu_B_mean,
                    'bsc/logE_mu_A_mean':    m._last_bsc_logE_mu_A_mean,
                    'bsc/logit_delta_mean':  m._last_bsc_logit_delta_mean,
                }, step=sccm.GLOBAL_STEP)
                return
    except Exception:
        pass


def _log_shc_diagnostics(model):
    """Walk modules to find SHCovisHead and log SH coefficient stats."""
    try:
        for m in model.modules():
            if hasattr(m, '_last_shc_coef_norm') and hasattr(m, 'L_max'):
                wandb.log({
                    'shc/coef_norm':         m._last_shc_coef_norm,
                    'shc/coef_dc':           m._last_shc_coef_dc,
                    'shc/high_band_energy':  m._last_shc_high_band_energy,
                }, step=sccm.GLOBAL_STEP)
                return
    except Exception:
        pass


def _log_dac_diagnostics(model):
    """Walk modules to find SphereCovisHead with dac_enabled and log β + d-proxy stats."""
    try:
        for m in model.modules():
            if getattr(m, 'dac_enabled', False) and hasattr(m, '_last_dac_beta'):
                wandb.log({
                    'dac/beta':              m._last_dac_beta,
                    'dac/dproxy_mean':       m._last_dac_dproxy_mean,
                    'dac/dproxy_std':        m._last_dac_dproxy_std,
                    'dac/logit_delta_mean':  m._last_dac_logit_delta_mean,
                }, step=sccm.GLOBAL_STEP)
                return
    except Exception:
        pass


def _log_pcc_diagnostics(model):
    """Walk modules to find SphereCovisMatcher with pcc_enabled and log γ + alignment stats."""
    try:
        for m in model.modules():
            if getattr(m, 'pcc_enabled', False) and hasattr(m, '_last_pcc_gamma'):
                wandb.log({
                    'pcc/gamma':             m._last_pcc_gamma,
                    'pcc/t_norm':            m._last_pcc_t_norm,
                    'pcc/align_mean':        m._last_pcc_align_mean,
                    'pcc/logit_delta_mean':  m._last_pcc_logit_delta_mean,
                }, step=sccm.GLOBAL_STEP)
                return
    except Exception:
        pass


def _log_gacfast_diagnostics(model):
    """Walk modules to find SphereCovisMatcher and log its GAC-fast stats."""
    try:
        for m in model.modules():
            if getattr(m, 'gacfast_enabled', False) and hasattr(m, '_last_gacfast_gamma'):
                wandb.log({
                    'gacfast/gamma_eff':     m._last_gacfast_gamma,
                    'gacfast/gamma_raw':     m._last_gacfast_gamma_raw,
                    'gacfast/d_geo_mean':    m._last_gacfast_d_geo_mean,
                    'gacfast/d_geo_max':     m._last_gacfast_d_geo_max,
                    'gacfast/warmup_factor': m._last_gacfast_warmup_factor,
                }, step=sccm.GLOBAL_STEP)
                return
    except Exception:
        pass


def _log_gacfine_diagnostics(model):
    """Walk modules to find Decoder and log GAC-fine per-scale stats."""
    try:
        for m in model.modules():
            if getattr(m, 'gacfine_enabled', False) and hasattr(m, '_last_gacfine'):
                last = m._last_gacfine
                if not last:
                    return
                payload = {}
                for scale_int, stats in last.items():
                    for k, v in stats.items():
                        payload[f'gacfine_s{scale_int}/{k}'] = v
                if payload:
                    wandb.log(payload, step=sccm.GLOBAL_STEP)
                return
    except Exception:
        pass


def train_step(train_batch, model, objective, optimizer, grad_scaler,
               grad_clip_norm = 1., grad_clip_by_substring=None, **kwargs):
    optimizer.zero_grad()
    out = model(train_batch)
    l = objective(out, train_batch)
    grad_scaler.scale(l).backward()
    grad_scaler.unscale_(optimizer)
    log_param_statistics(model.named_parameters())
    _apply_grad_clip(list(model.named_parameters()), grad_clip_norm, grad_clip_by_substring)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    wandb.log({"grad_scale": grad_scaler._scale.item()}, step = sccm.GLOBAL_STEP)
    _log_gacv2_diagnostics(model)
    _log_gacfine_diagnostics(model)
    _log_gacfast_diagnostics(model)
    _log_alac_diagnostics(model)
    _log_bsc_diagnostics(model)
    _log_shc_diagnostics(model)
    _log_dac_diagnostics(model)
    _log_pcc_diagnostics(model)
    if grad_scaler._scale < 1.:
        grad_scaler._scale = torch.tensor(1.).to(grad_scaler._scale)
    sccm.GLOBAL_STEP = sccm.GLOBAL_STEP + sccm.STEP_SIZE # increment global step
    return {"train_out": out, "train_loss": l.item()}


def train_k_steps(
    n_0, k, dataloader, model, objective, optimizer, lr_scheduler, grad_scaler, progress_bar=True, grad_clip_norm = 1., grad_clip_by_substring = None, warmup = None, ema_model = None, pbar_n_seconds = 1,
):
    for n in tqdm(range(n_0, n_0 + k), disable=(not progress_bar) or sccm.RANK > 0, mininterval=pbar_n_seconds):
        batch = next(dataloader)
        model.train(True)
        batch = to_cuda(batch)
        train_step(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_scaler=grad_scaler,
            n=n,
            grad_clip_norm = grad_clip_norm,
            grad_clip_by_substring = grad_clip_by_substring,
        )
        if ema_model is not None:
            ema_model.update()
        if warmup is not None:
            with warmup.dampening():
                lr_scheduler.step()
        else:
            lr_scheduler.step()
        [wandb.log({f"lr_group_{grp}": lr}, step=sccm.GLOBAL_STEP) for grp, lr in enumerate(lr_scheduler.get_last_lr())]


def train_epoch(
    dataloader=None,
    model=None,
    objective=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
):
    model.train(True)
    print(f"At epoch {epoch}")
    for batch in tqdm(dataloader, mininterval=5.0):
        batch = to_cuda(batch)
        train_step(
            train_batch=batch, model=model, objective=objective, optimizer=optimizer
        )
    lr_scheduler.step()
    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "epoch": epoch,
    }


def train_k_epochs(
    start_epoch, end_epoch, dataloader, model, objective, optimizer, lr_scheduler
):
    for epoch in range(start_epoch, end_epoch + 1):
        train_epoch(
            dataloader=dataloader,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
        )
