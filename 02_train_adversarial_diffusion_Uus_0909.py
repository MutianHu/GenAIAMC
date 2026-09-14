"""
Stage AD-2: Target-guided conditional adversarial diffusion attack generator.

Anti-saturation revision.

Why this revision exists
------------------------
The previous training setup used an unbounded negative-cross-entropy attack
objective together with strict tanh reparameterization.  Once the DDPM output
became large, tanh(raw_delta) approached +/-1 almost everywhere, the realized
perturbation power approached eps^2, and the tanh derivative became nearly zero.
The result was a frozen sign-like perturbation pattern whose validation ASR
barely changed between epochs.

This version keeps the newer/correct receiver-domain pipeline behavior while
borrowing the useful optimization behavior from the older script:

1. Default attack objective is the old self-limiting probability suppression
   loss -log(1 - p_y).  It stops pushing aggressively once p_y is already low.
2. The strict L_inf mapping remains tanh-based, but uses a temperature and is
   evaluated in FP32.
3. A direct raw-state anti-saturation barrier supplies gradient even if tanh is
   already close to flat.
4. The corrected OOB-energy penalty is retained.  A separate *small* near-DC /
   low-frequency concentration penalty replaces the accidental low-frequency
   penalty that existed in the older FFT implementation.
5. Detailed diagnostics report raw magnitude, tanh slope, saturation ratio,
   boundary occupancy, power/eps^2, low-frequency ratio, DC ratio and gradient
   norm after every epoch.
6. Validation noise is deterministic across epochs, so ASR changes mostly
   reflect learning rather than a different DDPM noise realization.

Important defaults for an 8 GB GPU
----------------------------------
- train batch: 8
- validation batch: 16
- AMP: OFF by default for diagnosis/stability.  Enable explicitly with --amp
  if needed after verifying that saturation remains controlled.
- diffusion steps, network size and epsilon schedule are unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pipeline_common_0909_fixed import (
    AdversarialConditionalNoisePredictor,
    ARTIFACT_ROOT,
    GlobalRMSNormalizer,
    INPUT_CLIP,
    TargetAMC,
    corpus_path,
    dataframe_iq,
    load_artifact,
    make_label_encoder,
    model_path,
    save_artifact,
    set_global_seed,
)


# =============================================================================
# Diffusion utilities
# =============================================================================


def extract(a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


def make_beta_schedule(steps: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, steps, device=device)


def predict_x0(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    a = extract(alpha_bar, t, x_t.shape)
    return (x_t - torch.sqrt(1 - a) * eps) / torch.sqrt(a)


def ddpm_step(
    model: torch.nn.Module,
    delta_t: torch.Tensor,
    clean: torch.Tensor,
    labels: torch.Tensor,
    t: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    predicted_noise = model(
        delta_t,
        clean,
        t.float() / len(betas),
        labels,
    )

    x0 = predict_x0(delta_t, predicted_noise, t, betas)
    alpha = extract(1.0 - betas, t, delta_t.shape)
    alpha_bar = extract(torch.cumprod(1.0 - betas, dim=0), t, delta_t.shape)

    if t[0] > 0:
        noise = torch.randn_like(delta_t)
        beta = extract(betas, t, delta_t.shape)
        mean = (
            1.0
            / torch.sqrt(alpha)
            * (
                delta_t
                - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise
            )
        )
        return mean + torch.sqrt(beta) * noise

    return x0


def generate_delta(
    model: torch.nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    """Differentiable DDPM reverse chain.

    Do not detach inside this loop during training.  The final attack objective
    remains differentiable through all reverse steps.
    """
    delta = torch.randn_like(clean)

    for i in reversed(range(len(betas))):
        t = torch.full(
            (clean.size(0),),
            i,
            device=clean.device,
            dtype=torch.long,
        )
        delta = ddpm_step(
            model,
            delta,
            clean,
            labels,
            t,
            betas,
        )

    return delta


# =============================================================================
# Strict perturbation bound and losses
# =============================================================================


def strict_tanh_bound(
    raw_delta: torch.Tensor,
    epsilon: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict L_inf reparameterization with a softer tanh input.

    z = raw_delta / temperature
    bounded_unit = tanh(z)
    delta = epsilon * bounded_unit

    A temperature > 1 keeps the tanh input in a region with useful derivative
    for longer while preserving the exact |delta| < epsilon bound.

    This mapping is always evaluated in FP32, even when --amp is enabled.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    raw_fp32 = raw_delta.float()
    squash_input = raw_fp32 / float(temperature)
    bounded_unit = torch.tanh(squash_input)
    delta = float(epsilon) * bounded_unit
    return squash_input, bounded_unit, delta


def attack_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    objective: str = "prob_suppression",
    margin: float = 0.2,
) -> torch.Tensor:
    """Untargeted attack objective minimized by the perturbation generator.

    prob_suppression (default)
        L = -log(1 - p_y)
        This is the useful behavior from the older script.  When the true-class
        probability p_y is already small, both the loss and its gradient become
        small.  It therefore does not keep rewarding arbitrarily extreme logits.

    cw_margin
        Stops contributing once a wrong-class logit exceeds the true-class
        logit by the requested margin.

    negative_ce
        Retained only for ablation.  Minimized -CE is unbounded below and keeps
        pushing after misclassification, making tanh saturation much more likely.
    """
    logits_fp32 = logits.float()

    if objective == "prob_suppression":
        probabilities = F.softmax(logits_fp32, dim=1)
        py = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        py = py.clamp(min=0.0, max=1.0 - 1e-6)
        return (-torch.log1p(-py)).mean()

    if objective == "cw_margin":
        true_logit = logits_fp32.gather(1, labels.unsqueeze(1)).squeeze(1)
        other_logits = logits_fp32.masked_fill(
            F.one_hot(labels, num_classes=logits_fp32.shape[1]).bool(),
            -torch.inf,
        )
        best_other = other_logits.max(dim=1).values
        return F.relu(true_logit - best_other + float(margin)).mean()

    if objective == "negative_ce":
        return -F.cross_entropy(logits_fp32, labels)

    raise ValueError(f"Unsupported attack objective: {objective}")


def power_loss(delta: torch.Tensor) -> torch.Tensor:
    return torch.mean(delta.float() ** 2)


def spectrum_oob_loss(
    delta: torch.Tensor,
    keep_ratio: float = 0.25,
) -> torch.Tensor:
    """Normalized out-of-band energy ratio for an unshifted baseband FFT.

    For an unshifted FFT, low absolute frequencies lie at both ends of the FFT
    vector, so the middle bins are treated as OOB.
    """
    if not 0.0 < keep_ratio <= 0.5:
        raise ValueError("keep_ratio must be in (0, 0.5]")

    freq = torch.fft.fft(delta.float(), dim=-1, norm="ortho")
    spec_power = torch.abs(freq) ** 2
    n = spec_power.shape[-1]
    keep = max(1, min(int(n * keep_ratio), n // 2))

    oob_mask = torch.zeros_like(spec_power)
    if keep < n - keep:
        oob_mask[..., keep:-keep] = 1.0

    total_energy = spec_power.sum(dim=(1, 2)).clamp_min(1e-12)
    oob_energy = (spec_power * oob_mask).sum(dim=(1, 2))
    return (oob_energy / total_energy).mean()


def low_frequency_concentration_loss(
    delta: torch.Tensor,
    low_ratio: float = 0.02,
) -> torch.Tensor:
    """Penalize collapse into DC / very-low-frequency perturbations.

    The older script accidentally penalized a much broader low-frequency region
    because of its FFT mask orientation.  That implementation was not a correct
    OOB penalty, but it did discourage DC-like collapse.  Here we retain the
    physically correct OOB penalty above and add only a small explicit near-DC
    concentration penalty.

    For an unshifted FFT, near-DC energy occupies the first bins and the final
    negative-frequency bins.
    """
    if not 0.0 < low_ratio < 0.5:
        raise ValueError("low_ratio must be in (0, 0.5)")

    freq = torch.fft.fft(delta.float(), dim=-1, norm="ortho")
    spec_power = torch.abs(freq) ** 2
    n = spec_power.shape[-1]
    low_bins = max(1, min(int(n * low_ratio), n // 2))

    low_energy = (
        spec_power[..., :low_bins].sum(dim=(1, 2))
        + spec_power[..., -low_bins:].sum(dim=(1, 2))
    )
    total_energy = spec_power.sum(dim=(1, 2)).clamp_min(1e-12)
    return (low_energy / total_energy).mean()


def saturation_barrier_loss(
    squash_input: torch.Tensor,
    squash_limit: float = 1.5,
) -> torch.Tensor:
    """Direct anti-saturation barrier before tanh.

    This loss acts directly on z = raw_delta / temperature rather than on the
    bounded perturbation.  Therefore it still supplies a useful gradient when
    tanh(z) is already close to flat.

    tanh(1.5) ~= 0.905 and its local slope is still about 0.18.
    """
    excess = F.relu(squash_input.float().abs() - float(squash_limit))
    return torch.mean(excess ** 2)


def dc_ratio_metric(delta: torch.Tensor) -> torch.Tensor:
    """Normalized time-domain DC power ratio, used only as a diagnostic."""
    delta_fp32 = delta.float()
    dc_power = delta_fp32.mean(dim=-1).pow(2)
    total_power = delta_fp32.pow(2).mean(dim=-1).clamp_min(1e-12)
    return (dc_power / total_power).mean()


# =============================================================================
# Diagnostics
# =============================================================================


def perturbation_diagnostics(
    raw_delta: torch.Tensor,
    squash_input: torch.Tensor,
    bounded_unit: torch.Tensor,
    realized_delta: torch.Tensor,
    epsilon: float,
    saturation_threshold: float,
    low_frequency_ratio: float,
) -> dict[str, float]:
    with torch.no_grad():
        raw = raw_delta.detach().float()
        z = squash_input.detach().float()
        bounded = bounded_unit.detach().float()
        delta = realized_delta.detach().float()

        abs_raw = raw.abs()
        abs_z = z.abs()
        abs_bounded = bounded.abs()
        tanh_slope = 1.0 - bounded.pow(2)

        threshold = float(saturation_threshold)
        sat_ratio = (abs_bounded >= threshold).float().mean()

        if epsilon > 0:
            boundary_ratio = (
                delta.abs() >= threshold * float(epsilon)
            ).float().mean()
            power_ratio = delta.pow(2).mean() / (float(epsilon) ** 2)
        else:
            boundary_ratio = torch.tensor(0.0, device=delta.device)
            power_ratio = torch.tensor(0.0, device=delta.device)

        low_ratio = low_frequency_concentration_loss(
            delta,
            low_ratio=low_frequency_ratio,
        )
        dc_ratio = dc_ratio_metric(delta)

        return {
            "raw_abs_mean": float(abs_raw.mean().item()),
            "raw_abs_max": float(abs_raw.max().item()),
            "squash_abs_mean": float(abs_z.mean().item()),
            "squash_abs_max": float(abs_z.max().item()),
            "bounded_abs_mean": float(abs_bounded.mean().item()),
            "tanh_slope_mean": float(tanh_slope.mean().item()),
            "saturation_ratio": float(sat_ratio.item()),
            "boundary_ratio": float(boundary_ratio.item()),
            "power_to_eps2": float(power_ratio.item()),
            "low_frequency_ratio": float(low_ratio.item()),
            "dc_ratio": float(dc_ratio.item()),
        }


# =============================================================================
# Dataset
# =============================================================================


class CleanDataset(Dataset):
    def __init__(self, dataframe, normalizer):
        self.x = normalizer.transform(dataframe_iq(dataframe))
        self.y = dataframe["Modulation_Label"].values
        encoder = make_label_encoder()
        self.labels = encoder.transform(self.y)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# =============================================================================
# RNG helpers for deterministic validation
# =============================================================================


def capture_rng_state(device: torch.device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state_all()
    return cpu_state, cuda_state


def restore_rng_state(device: torch.device, state) -> None:
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


# =============================================================================
# Validation
# =============================================================================


def evaluate_attack(
    diffusion: torch.nn.Module,
    amc: torch.nn.Module,
    val_loader: DataLoader,
    betas: torch.Tensor,
    device: torch.device,
    val_eps_list: list[float],
    validation_seed: int,
    use_amp: bool,
    tanh_temperature: float,
    saturation_threshold: float,
) -> dict[str, Any]:
    diffusion.eval()

    successful_counts = [0 for _ in val_eps_list]
    clean_correct_count = 0
    total_count = 0

    sat_count = 0
    element_count = 0
    tanh_slope_sum = 0.0
    bounded_abs_sum = 0.0

    rng_state = capture_rng_state(device)
    torch.manual_seed(validation_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(validation_seed)

    try:
        with torch.no_grad():
            for clean, labels in val_loader:
                clean = clean.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                clean_fp32 = clean.float()

                with autocast_context(use_amp):
                    clean_logits = amc(clean_fp32)
                clean_preds = clean_logits.argmax(dim=1)
                clean_correct = clean_preds == labels

                clean_correct_count += int(clean_correct.sum().item())
                total_count += int(labels.numel())

                with autocast_context(use_amp):
                    raw_delta = generate_delta(
                        diffusion,
                        clean,
                        labels,
                        betas,
                    )

                squash_input, bounded_unit, _ = strict_tanh_bound(
                    raw_delta,
                    epsilon=1.0,
                    temperature=tanh_temperature,
                )
                del squash_input

                sat_count += int(
                    (bounded_unit.abs() >= saturation_threshold).sum().item()
                )
                element_count += int(bounded_unit.numel())
                tanh_slope_sum += float(
                    (1.0 - bounded_unit.pow(2)).sum().item()
                )
                bounded_abs_sum += float(
                    bounded_unit.abs().sum().item()
                )

                # Use the same stochastic direction for every epsilon so the
                # validation curve isolates budget scaling.
                for eps_idx, eval_eps in enumerate(val_eps_list):
                    eval_delta = float(eval_eps) * bounded_unit
                    eval_adv = torch.clamp(
                        clean_fp32 + eval_delta,
                        -INPUT_CLIP,
                        INPUT_CLIP,
                    )

                    with autocast_context(use_amp):
                        adv_logits = amc(eval_adv)
                    adv_preds = adv_logits.argmax(dim=1)

                    successful = clean_correct & (adv_preds != labels)
                    successful_counts[eps_idx] += int(
                        successful.sum().item()
                    )
    finally:
        restore_rng_state(device, rng_state)

    val_asr_scores = [
        successful / max(clean_correct_count, 1)
        for successful in successful_counts
    ]

    return {
        "asr_scores": val_asr_scores,
        "clean_accuracy": clean_correct_count / max(total_count, 1),
        "saturation_ratio": sat_count / max(element_count, 1),
        "tanh_slope_mean": tanh_slope_sum / max(element_count, 1),
        "bounded_abs_mean": bounded_abs_sum / max(element_count, 1),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser()

    # The older script used about 6 epochs per curriculum stage.  Keep the newer
    # monotonic 5-stage schedule, but give each stage enough time to adapt.
    parser.add_argument("--epochs", type=int, default=15)

    # Conservative full-FP32 default for an 8 GB GPU.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--validation-seed", type=int, default=3040)

    # AMP is now opt-in rather than automatic.  This makes the first diagnostic
    # run maximally comparable and avoids FP16 numerical saturation as a confound.
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA AMP for DDPM/AMC forward passes. Default: off.",
    )

    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=(
            ARTIFACT_ROOT
            / "models/adversarial_diffusion_unsupervised_0909.pt"
        ),
    )

    # Loss weights.
    parser.add_argument("--lambda-power", type=float, default=0.01)
    parser.add_argument("--lambda-frequency", type=float, default=0.01)
    parser.add_argument("--lambda-low-frequency", type=float, default=0.05)
    parser.add_argument("--lambda-saturation", type=float, default=0.05)

    parser.add_argument("--frequency-keep-ratio", type=float, default=0.25)
    parser.add_argument("--low-frequency-ratio", type=float, default=0.02)

    # Anti-saturation controls.
    parser.add_argument("--tanh-temperature", type=float, default=2.0)
    parser.add_argument("--squash-limit", type=float, default=1.5)
    parser.add_argument("--saturation-threshold", type=float, default=0.98)

    parser.add_argument(
        "--attack-objective",
        choices=["prob_suppression", "cw_margin", "negative_ce"],
        default="prob_suppression",
        help=(
            "Default restores the older script's self-limiting -log(1-p_y) "
            "objective. negative_ce is retained only for ablation."
        ),
    )
    parser.add_argument(
        "--attack-margin",
        type=float,
        default=0.2,
        help="Wrong-class logit margin for --attack-objective=cw_margin.",
    )

    parser.add_argument("--data-ratio", type=float, default=0.5)

    # Keep the newer monotonic curriculum because the generator is not epsilon-
    # conditioned.  Returning to larger eps later is therefore not required.
    parser.add_argument(
        "--eps-schedule",
        nargs="+",
        type=float,
        default=[0.3, 0.2, 0.1, 0.05, 0.03],
        help="Large-to-small curriculum, e.g. 0.3 0.2 0.1 0.05 0.03.",
    )

    parser.add_argument(
        "--validation-eps-list",
        nargs="+",
        type=float,
        default=[0.03, 0.05, 0.10, 0.20, 0.30],
    )

    parser.add_argument(
        "--validation-eps-weighting",
        choices=["uniform", "inverse_eps"],
        default="inverse_eps",
        help=(
            "inverse_eps gives small perturbation budgets more influence during "
            "checkpoint selection."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Argument validation
    # -------------------------------------------------------------------------
    if not args.eps_schedule or any(eps <= 0 for eps in args.eps_schedule):
        parser.error("--eps-schedule must contain positive values")

    if any(
        later > earlier
        for earlier, later in zip(args.eps_schedule, args.eps_schedule[1:])
    ):
        parser.error("--eps-schedule must be monotonically non-increasing")

    if args.epochs < len(args.eps_schedule):
        parser.error("--epochs must be at least the number of epsilon stages")

    if not args.validation_eps_list or any(
        eps <= 0 for eps in args.validation_eps_list
    ):
        parser.error("--validation-eps-list must contain positive values")

    if not 0.0 < args.data_ratio <= 1.0:
        parser.error("--data-ratio must be in (0, 1]")

    if not 0.0 < args.frequency_keep_ratio <= 0.5:
        parser.error("--frequency-keep-ratio must be in (0, 0.5]")

    if not 0.0 < args.low_frequency_ratio < args.frequency_keep_ratio:
        parser.error(
            "--low-frequency-ratio must be positive and smaller than "
            "--frequency-keep-ratio"
        )

    if args.batch_size <= 0 or args.val_batch_size <= 0:
        parser.error("batch sizes must be positive")

    if args.tanh_temperature <= 0:
        parser.error("--tanh-temperature must be positive")

    if args.squash_limit <= 0:
        parser.error("--squash-limit must be positive")

    if not 0.0 < args.saturation_threshold < 1.0:
        parser.error("--saturation-threshold must be in (0, 1)")

    for name in (
        "lambda_power",
        "lambda_frequency",
        "lambda_low_frequency",
        "lambda_saturation",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    if args.amp and device.type != "cuda":
        print("[WARN] --amp requested but CUDA is unavailable; AMP disabled.")

    # -------------------------------------------------------------------------
    # Frozen TargetAMC + its own normalizer
    # -------------------------------------------------------------------------
    checkpoint = load_artifact(model_path("target"))

    amc = TargetAMC().to(device)
    amc.load_state_dict(checkpoint["model_state"])
    amc.eval()
    for parameter in amc.parameters():
        parameter.requires_grad = False

    if "normalizer" in checkpoint:
        normalizer = GlobalRMSNormalizer.from_state_dict(
            checkpoint["normalizer"]
        )
    else:
        target_corpus = load_artifact(corpus_path("target"))
        normalizer = GlobalRMSNormalizer.fit(
            target_corpus["splits"]["train"]
        )
        print(
            "[WARN] TargetAMC checkpoint has no normalizer; fitted one from "
            "C-Target/train. Retraining TargetAMC with a saved normalizer is "
            "recommended."
        )

    # -------------------------------------------------------------------------
    # Attack-training data
    # -------------------------------------------------------------------------
    corpus = load_artifact(corpus_path("diff"))
    train_df = corpus["splits"]["train"]

    if args.data_ratio < 1.0:
        print(
            f"[INFO] 快速测试模式：仅使用 {args.data_ratio * 100:.0f}% 的训练数据。"
        )
        train_df = train_df.sample(
            frac=args.data_ratio,
            random_state=args.seed,
        ).reset_index(drop=True)

    train_set = CleanDataset(train_df, normalizer)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # -------------------------------------------------------------------------
    # Fixed balanced validation subset
    # -------------------------------------------------------------------------
    val_df = corpus["splits"]["val"]
    print("[INFO] 抽取固定验证集进行每轮评估 (保证各调制类别完全均衡)...")

    fixed_val_df = val_df.groupby(
        "Modulation_Label",
        group_keys=False,
    ).apply(
        lambda x: x.sample(n=64, random_state=args.seed)
    ).reset_index(drop=True)

    val_set = CleanDataset(fixed_val_df, normalizer)
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
    )

    # -------------------------------------------------------------------------
    # Diffusion attacker
    # -------------------------------------------------------------------------
    diffusion = AdversarialConditionalNoisePredictor(num_classes=6).to(device)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)
    betas = make_beta_schedule(args.diffusion_steps, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(
        f"[INFO] device={device}, train_batch={args.batch_size}, "
        f"val_batch={args.val_batch_size}, AMP={'ON' if use_amp else 'OFF'}"
    )
    print(
        "[INFO] anti-saturation: "
        f"objective={args.attack_objective}, "
        f"tanh_temperature={args.tanh_temperature:g}, "
        f"squash_limit={args.squash_limit:g}, "
        f"lambda_sat={args.lambda_saturation:g}, "
        f"lambda_lowfreq={args.lambda_low_frequency:g}"
    )

    if args.attack_objective == "negative_ce":
        print(
            "[WARN] negative_ce is an ablation setting. It keeps pushing after "
            "misclassification and has a higher tanh-saturation risk."
        )

    # -------------------------------------------------------------------------
    # Curriculum + checkpoint bookkeeping
    # -------------------------------------------------------------------------
    best_val_attack_score = -float("inf")
    best_state = None
    best_epoch = None
    best_training_eps = None
    diagnostic_history: list[dict[str, Any]] = []

    num_stages = len(args.eps_schedule)
    epochs_per_stage = max(1, args.epochs // num_stages)
    active_eps = args.eps_schedule[0]

    print("[INFO] 开启课程学习 (Curriculum Learning) 模式")
    print(
        "[INFO] 约束策略: FP32 Temperature-Tanh + Raw-State Anti-Saturation Barrier"
    )
    print(f"[INFO] 训练阶段分配: 每 {epochs_per_stage} 个 Epoch 增加一次难度")
    print(f"[INFO] 难度路线图: eps = {args.eps_schedule}")

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        stage_idx = min(epoch // epochs_per_stage, num_stages - 1)
        new_eps = args.eps_schedule[stage_idx]

        if new_eps != active_eps:
            print(
                f"\n[CURRICULUM] eps constraint tightened: "
                f"{active_eps} -> {new_eps}"
            )
            active_eps = new_eps

        diffusion.train()

        metric_lists: dict[str, list[float]] = {
            "loss": [],
            "attack": [],
            "power": [],
            "oob": [],
            "lowfreq": [],
            "sat_loss": [],
            "raw_abs_mean": [],
            "raw_abs_max": [],
            "squash_abs_mean": [],
            "squash_abs_max": [],
            "bounded_abs_mean": [],
            "tanh_slope": [],
            "sat_ratio": [],
            "boundary_ratio": [],
            "power_to_eps2": [],
            "dc_ratio": [],
            "grad_norm": [],
        }

        for clean, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1} (eps={active_eps})",
        ):
            clean = clean.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean_fp32 = clean.float()

            optimizer.zero_grad(set_to_none=True)

            # The expensive DDPM chain may optionally use AMP.  The strict tanh
            # mapping and all regularizers stay FP32.
            with autocast_context(use_amp):
                raw_delta = generate_delta(
                    diffusion,
                    clean,
                    labels,
                    betas,
                )

            squash_input, bounded_unit, candidate_delta = strict_tanh_bound(
                raw_delta,
                epsilon=active_eps,
                temperature=args.tanh_temperature,
            )

            adv = torch.clamp(
                clean_fp32 + candidate_delta,
                -INPUT_CLIP,
                INPUT_CLIP,
            )
            delta = adv - clean_fp32

            # Frozen AMC parameters do not receive gradients, but gradient wrt
            # adv must remain available for the attacker.
            with autocast_context(use_amp):
                logits = amc(adv)

            loss_attack = attack_loss(
                logits,
                labels,
                objective=args.attack_objective,
                margin=args.attack_margin,
            )
            loss_power = power_loss(delta)
            loss_oob = spectrum_oob_loss(
                delta,
                keep_ratio=args.frequency_keep_ratio,
            )
            loss_lowfreq = low_frequency_concentration_loss(
                delta,
                low_ratio=args.low_frequency_ratio,
            )
            loss_sat = saturation_barrier_loss(
                squash_input,
                squash_limit=args.squash_limit,
            )

            loss = (
                loss_attack
                + args.lambda_power * loss_power
                + args.lambda_frequency * loss_oob
                + args.lambda_low_frequency * loss_lowfreq
                + args.lambda_saturation * loss_sat
            )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Returned norm is the total norm BEFORE clipping.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                diffusion.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            diag = perturbation_diagnostics(
                raw_delta=raw_delta,
                squash_input=squash_input,
                bounded_unit=bounded_unit,
                realized_delta=delta,
                epsilon=active_eps,
                saturation_threshold=args.saturation_threshold,
                low_frequency_ratio=args.low_frequency_ratio,
            )

            metric_lists["loss"].append(float(loss.detach().item()))
            metric_lists["attack"].append(float(loss_attack.detach().item()))
            metric_lists["power"].append(float(loss_power.detach().item()))
            metric_lists["oob"].append(float(loss_oob.detach().item()))
            metric_lists["lowfreq"].append(float(loss_lowfreq.detach().item()))
            metric_lists["sat_loss"].append(float(loss_sat.detach().item()))
            metric_lists["raw_abs_mean"].append(diag["raw_abs_mean"])
            metric_lists["raw_abs_max"].append(diag["raw_abs_max"])
            metric_lists["squash_abs_mean"].append(diag["squash_abs_mean"])
            metric_lists["squash_abs_max"].append(diag["squash_abs_max"])
            metric_lists["bounded_abs_mean"].append(diag["bounded_abs_mean"])
            metric_lists["tanh_slope"].append(diag["tanh_slope_mean"])
            metric_lists["sat_ratio"].append(diag["saturation_ratio"])
            metric_lists["boundary_ratio"].append(diag["boundary_ratio"])
            metric_lists["power_to_eps2"].append(diag["power_to_eps2"])
            metric_lists["dc_ratio"].append(diag["dc_ratio"])

            grad_norm_value = float(
                grad_norm.detach().item()
                if torch.is_tensor(grad_norm)
                else grad_norm
            )
            metric_lists["grad_norm"].append(grad_norm_value)

        epoch_metrics = {
            key: float(np.mean(values))
            for key, values in metric_lists.items()
            if key not in {"raw_abs_max", "squash_abs_max"}
        }
        epoch_metrics["raw_abs_max"] = float(
            np.max(metric_lists["raw_abs_max"])
        )
        epoch_metrics["squash_abs_max"] = float(
            np.max(metric_lists["squash_abs_max"])
        )
        epoch_grad_norm_max = float(np.max(metric_lists["grad_norm"]))

        # ---------------------------------------------------------------------
        # Deterministic validation
        # ---------------------------------------------------------------------
        val_result = evaluate_attack(
            diffusion=diffusion,
            amc=amc,
            val_loader=val_loader,
            betas=betas,
            device=device,
            val_eps_list=args.validation_eps_list,
            validation_seed=args.validation_seed,
            use_amp=use_amp,
            tanh_temperature=args.tanh_temperature,
            saturation_threshold=args.saturation_threshold,
        )
        diffusion.train()

        val_asr_scores = val_result["asr_scores"]

        if args.validation_eps_weighting == "inverse_eps":
            val_weights = 1.0 / np.asarray(
                args.validation_eps_list,
                dtype=np.float64,
            )
        else:
            val_weights = np.ones(
                len(args.validation_eps_list),
                dtype=np.float64,
            )

        val_attack_score = float(
            np.average(
                np.asarray(val_asr_scores, dtype=np.float64),
                weights=val_weights,
            )
        )

        val_asr_text = ", ".join(
            f"eps={eps:g}:{asr:.3f}"
            for eps, asr in zip(args.validation_eps_list, val_asr_scores)
        )

        if val_attack_score > best_val_attack_score:
            best_val_attack_score = val_attack_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in diffusion.state_dict().items()
            }
            best_epoch = epoch + 1
            best_training_eps = float(active_eps)
            best_text = " BEST"
        else:
            best_text = ""

        record: dict[str, Any] = {
            "epoch": int(epoch + 1),
            "training_eps": float(active_eps),
            **epoch_metrics,
            "grad_norm_max": epoch_grad_norm_max,
            "val_clean_accuracy": float(val_result["clean_accuracy"]),
            "val_saturation_ratio": float(val_result["saturation_ratio"]),
            "val_tanh_slope_mean": float(val_result["tanh_slope_mean"]),
            "val_bounded_abs_mean": float(val_result["bounded_abs_mean"]),
            "val_attack_score": val_attack_score,
            "val_asr_scores": [float(x) for x in val_asr_scores],
        }
        diagnostic_history.append(record)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={epoch_metrics['loss']:.6f} "
            f"attack={epoch_metrics['attack']:.6f} "
            f"power={epoch_metrics['power']:.6f} "
            f"oob={epoch_metrics['oob']:.6f} "
            f"lowfreq={epoch_metrics['lowfreq']:.6f} "
            f"sat_loss={epoch_metrics['sat_loss']:.6f} "
            f"VAL_ASR_SCORE={val_attack_score:.4f}{best_text} | "
            f"{val_asr_text}"
        )

        print(
            "[DIAG] "
            f"eps={active_eps:g} "
            f"raw|x|mean={epoch_metrics['raw_abs_mean']:.4f} "
            f"raw|max={epoch_metrics['raw_abs_max']:.4f} "
            f"|z|mean={epoch_metrics['squash_abs_mean']:.4f} "
            f"|z|max={epoch_metrics['squash_abs_max']:.4f} "
            f"|tanh|mean={epoch_metrics['bounded_abs_mean']:.4f} "
            f"tanh_slope={epoch_metrics['tanh_slope']:.4f} "
            f"sat_ratio={epoch_metrics['sat_ratio']:.2%} "
            f"boundary_ratio={epoch_metrics['boundary_ratio']:.2%} "
            f"power/eps^2={epoch_metrics['power_to_eps2']:.4f} "
            f"lowfreq_ratio={epoch_metrics['lowfreq']:.4f} "
            f"dc_ratio={epoch_metrics['dc_ratio']:.4f} "
            f"grad_norm={epoch_metrics['grad_norm']:.4f} "
            f"(max={epoch_grad_norm_max:.4f})"
        )

        print(
            "[VAL_DIAG] "
            f"clean_acc={val_result['clean_accuracy']:.4f} "
            f"sat_ratio={val_result['saturation_ratio']:.2%} "
            f"|tanh|mean={val_result['bounded_abs_mean']:.4f} "
            f"tanh_slope={val_result['tanh_slope_mean']:.4f}"
        )

        # Actionable warnings.
        if (
            epoch_metrics["sat_ratio"] > 0.20
            or epoch_metrics["tanh_slope"] < 0.20
        ):
            print(
                "[WARN] Tanh saturation is still high. First increase "
                "--lambda-saturation (e.g. 0.05 -> 0.1) or increase "
                "--tanh-temperature (e.g. 2 -> 3)."
            )

        if epoch_metrics["power_to_eps2"] > 0.80:
            print(
                "[WARN] perturbation power is close to eps^2; too many "
                "coordinates are sitting near the L_inf boundary."
            )

        if epoch_metrics["lowfreq"] > 0.50:
            print(
                "[WARN] perturbation energy is strongly concentrated near DC. "
                "Consider increasing --lambda-low-frequency."
            )

        if (
            not np.isfinite(epoch_metrics["grad_norm"])
            or epoch_metrics["grad_norm"] < 1e-8
        ):
            print(
                "[WARN] gradient norm is non-finite or nearly zero; training "
                "may be stalled."
            )

    # -------------------------------------------------------------------------
    # Restore best model and save checkpoint
    # -------------------------------------------------------------------------
    if best_state is not None:
        diffusion.load_state_dict(best_state)

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    save_artifact(
        {
            "model_state": diffusion.cpu().state_dict(),
            "normalizer": normalizer.state_dict(),
            "diffusion_steps": args.diffusion_steps,
            "seed": args.seed,
            "validation_seed": args.validation_seed,
            "batch_size": args.batch_size,
            "validation_batch_size": args.val_batch_size,
            "amp_enabled": use_amp,
            "training": (
                "target_guided_class_conditional_attack_"
                "curriculum_temperature_tanh_antisaturation_v2"
            ),
            "normalizer_role": "target_amc",
            "attack_objective": args.attack_objective,
            "attack_margin": args.attack_margin,
            "lambda_power": args.lambda_power,
            "lambda_frequency": args.lambda_frequency,
            "lambda_low_frequency": args.lambda_low_frequency,
            "lambda_saturation": args.lambda_saturation,
            "frequency_keep_ratio": args.frequency_keep_ratio,
            "low_frequency_ratio": args.low_frequency_ratio,
            "tanh_temperature": args.tanh_temperature,
            "squash_limit": args.squash_limit,
            "saturation_threshold": args.saturation_threshold,
            "frequency_loss": "normalized_out_of_band_energy_ratio",
            "low_frequency_loss": (
                "normalized_near_dc_energy_concentration_ratio"
            ),
            "saturation_loss": (
                "pre_tanh_squash_input_excess_squared_barrier"
            ),
            "best_val_attack_score": best_val_attack_score,
            "best_epoch": best_epoch,
            "best_training_eps": best_training_eps,
            "training_eps_schedule": args.eps_schedule,
            "validation_eps_list": args.validation_eps_list,
            "validation_eps_weighting": args.validation_eps_weighting,
            "final_eps": args.eps_schedule[-1],
            "diagnostic_history": diagnostic_history,
        },
        args.output_checkpoint,
    )

    print(
        "Saved anti-saturation adversarial diffusion model "
        f"to {args.output_checkpoint} "
        f"(best_epoch={best_epoch}, best_eps={best_training_eps}, "
        f"best_val_score={best_val_attack_score:.4f})."
    )


if __name__ == "__main__":
    main()
