"""
Stage AD-5T (0909): Target-source OFFLINE adversarial hardening of SurrogateAMC.

Research question
=================
Can adversarial knowledge learned/crafted on one fixed source AMC transfer into
robustness for a DIFFERENT defense AMC through a fixed offline adversarial corpus?

This script implements the controlled comparison requested in the paper:

    Fixed SourceAMC = Original TargetAMC
              |
              +--> Target-guided pretrained Diffusion --> fixed D_Diff
              |
              +--> PGD-10(TargetAMC)                 --> fixed D_PGD

    SAME original SurrogateAMC checkpoint
              |
              +--> Clean-Finetune control (optional/default)
              +--> Diffusion-Offline hardening with D_Diff
              +--> PGD-Offline hardening with D_PGD

The final robust models are SurrogateAMC models, NOT TargetAMC models.

Full-epoch training and loss logging
====================================
All selected defense models are trained for the complete --epochs budget (default 50).
Early stopping is intentionally disabled so every epoch is preserved for publication
curves.  Model selection is performed only AFTER the full run by the unchanged fixed
validation robustness score.  Per epoch we save: training CE, clean-validation CE,
fixed Target-PGD CE, fixed Target-Diffusion CE, their weighted diagnostic CE, and
accuracy/robustness metrics.  Clean-validation CE is the preferred cross-method loss
comparison because every method is evaluated on exactly the same clean validation set.

Core fairness invariants
========================
1) Diffusion and PGD share the SAME source AMC knowledge: Original TargetAMC.
2) Both attacks use the SAME TargetAMC normalizer / epsilon coordinate.
3) Both attack the SAME C-Diff/train rows with the SAME labels.
4) PGD stores one adversarial variant per source row. Diffusion diversity banks may
   store multiple variants, but cycle training exposes exactly N pairs per epoch so
   the optimizer-update budget remains comparable across PGD and Diffusion methods.
5) All defense models start from the SAME original SurrogateAMC checkpoint.
6) Defense training uses the SAME SurrogateAMC normalizer, optimizer, batch order,
   clean/adversarial ratio, epoch budget, and checkpoint rule.
7) NO adversarial example is generated against the moving SurrogateAMC during
   training or checkpoint selection.
8) Current-defense white-box FGSM/BIM/PGD is used ONLY in final held-out C-Eval.

Why this is a transfer-hardening experiment
===========================================
The source attacker sees TargetAMC, while the hardened model is SurrogateAMC.
Therefore both D_Diff and D_PGD are transfer adversarial corpora with respect to
SurrogateAMC.  PGD is NOT online PGD-AT here.

Diffusion provenance
====================
The existing Target-guided diffusion checkpoint is intentionally reused.  The
script verifies, as far as checkpoint metadata permits, that it was guided by
TargetAMC and that its stored normalizer matches TargetAMC.  A checkpoint that
explicitly records a non-target guide model is rejected.

Diffusion inference consistency
===============================
The provided AD-2 anti-saturation training uses

    delta = eps * tanh(raw_delta / tanh_temperature).

This script reads tanh_temperature from the attacker checkpoint and uses the
same mapping for offline generation, fixed validation, and final epsilon sweep.
This fixes the historical AD-3/AD-7R mismatch where tanh(raw_delta) was used
without the saved temperature.

Checkpoint selection
====================
Checkpoint selection deliberately DOES NOT generate PGD against the moving
SurrogateAMC.  It uses only:
    - clean validation accuracy;
    - robustness to a FIXED TargetAMC-PGD validation cache;
    - robustness to a FIXED Target-guided-Diffusion validation cache.
Thus the hardening/model-selection pipeline does not adapt attacks to the
current defense model.

Final evaluation
================
On one common held-out C-Eval subset:
    - clean accuracy;
    - frozen Target-trained PureDiffusion transfer;
    - TargetAMC-crafted FGSM/BIM/PGD transfer;
    - CURRENT-defense white-box FGSM/BIM/PGD for every evaluated SurrogateAMC;
    - epsilon sweep and per-SNR metrics;
    - perturbation L_inf/L2/power/PSR/OBER.

Dependencies beside this script
===============================
- pipeline_common_0909_fixed.py
- 03_adversarial_training_diffusion_0909.py
- 07_evaluate_at_robustness_only_0909.py

Use --training-script / --evaluator-script if your filenames differ.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pipeline_common_0909_fixed as common


# =============================================================================
# Dynamic imports
# =============================================================================


def _import_module(path: Path, name: str):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_script(
    explicit: Optional[Path],
    candidates: Sequence[str],
    label: str,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path

    here = Path(__file__).resolve().parent
    for name in candidates:
        path = here / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {label} beside this script. Pass its path explicitly."
    )


def resolve_training_script(explicit: Optional[Path]) -> Path:
    return resolve_script(
        explicit,
        [
            "03_adversarial_training_diffusion_0909.py",
            "03_adversarial_training_diffusion_0909_auto_remaining.py",
            "03_adversarial_training_diffusion_0909_final10pct.py",
            "03_adversarial_training_diffusion_0909(20260912-025638).py",
        ],
        "Stage AD-3 training utility script",
    )


def resolve_evaluator_script(explicit: Optional[Path]) -> Path:
    return resolve_script(
        explicit,
        ["07_evaluate_at_robustness_only_0909.py"],
        "Stage AD-7R detector-free evaluator",
    )


# =============================================================================
# General helpers
# =============================================================================


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def clean_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def clone_target(checkpoint: dict, device: torch.device) -> nn.Module:
    model = common.TargetAMC().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def clone_surrogate(checkpoint: dict, device: torch.device) -> nn.Module:
    model = common.SurrogateAMC().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def load_role_normalizer(
    checkpoint: dict,
    corpus_role: str,
    warning_label: str,
) -> common.GlobalRMSNormalizer:
    if "normalizer" in checkpoint:
        return common.GlobalRMSNormalizer.from_state_dict(checkpoint["normalizer"])

    corpus = common.load_artifact(common.corpus_path(corpus_role))
    normalizer = common.GlobalRMSNormalizer.fit(corpus["splits"]["train"])
    del corpus
    print(
        f"[WARN] {warning_label} checkpoint has no normalizer; "
        f"fitted one from C-{corpus_role.capitalize()}/train."
    )
    return normalizer


def normalize_physical(
    physical: torch.Tensor,
    normalizer: common.GlobalRMSNormalizer,
) -> torch.Tensor:
    return torch.clamp(
        physical / float(normalizer.scale),
        -common.INPUT_CLIP,
        common.INPUT_CLIP,
    )


def eps_tag(eps: float) -> str:
    return f"{float(eps):g}".replace("-", "m").replace(".", "p")


def split_calibration_and_validation(
    val_df: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified deterministic half split; second half is robust validation."""
    group_cols = [
        c for c in ["Modulation_Label", "SNR_dB"]
        if c in val_df.columns
    ]
    rng = np.random.default_rng(seed)
    left: List[int] = []
    right: List[int] = []

    grouped = (
        val_df.groupby(group_cols, sort=False, dropna=False)
        if group_cols
        else [(None, val_df)]
    )
    for _, group in grouped:
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = max(1, len(idx) // 2)
        left.extend(idx[:cut].tolist())
        right.extend(idx[cut:].tolist())

    left_df = val_df.loc[sorted(left)].reset_index(drop=True)
    right_df = val_df.loc[sorted(right)].reset_index(drop=True)
    if right_df.empty:
        raise RuntimeError("Robust validation split became empty.")
    return left_df, right_df


def stratified_source_sample(
    df: pd.DataFrame,
    ratio: float,
    seed: int,
) -> pd.DataFrame:
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0,1]")

    work = (
        df.reset_index(drop=False)
        .rename(columns={"index": "_Original_Row_Index"})
    )
    if ratio >= 1:
        return work.reset_index(drop=True)

    group_cols = [
        c for c in ["Modulation_Label", "SNR_dB"]
        if c in work.columns
    ]
    if not group_cols:
        n = max(1, int(round(len(work) * ratio)))
        return work.sample(n=n, random_state=seed).reset_index(drop=True)

    pieces = []
    for gi, (_, group) in enumerate(
        work.groupby(group_cols, sort=True, dropna=False)
    ):
        n = max(1, int(round(len(group) * ratio)))
        n = min(n, len(group))
        pieces.append(
            group.sample(n=n, random_state=seed + 7919 * gi)
        )

    return (
        pd.concat(pieces, axis=0)
        .sample(frac=1.0, random_state=seed + 104729)
        .reset_index(drop=True)
    )


# =============================================================================
# Datasets
# =============================================================================


class IndexedPhysicalDataset(Dataset):
    """Physical IQ + label/SNR + stable local index used to write mmap caches."""

    def __init__(self, df: pd.DataFrame, encoder):
        self.iq = common.dataframe_iq(df).astype(np.float32, copy=False)
        self.labels = encoder.transform(
            df["Modulation_Label"].astype(str).to_numpy()
        ).astype(np.int64)
        self.snrs = df["SNR_dB"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.iq[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.snrs[idx], dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long),
        )


class OfflinePairDataset(Dataset):
    """Physical clean IQ paired with one fixed physical adversarial corpus."""

    def __init__(
        self,
        clean_iq: np.ndarray,
        labels: np.ndarray,
        adv_path: Optional[Path],
    ):
        self.clean_iq = clean_iq
        self.labels = labels
        self.adv_path = adv_path
        self.adv_iq = (
            None if adv_path is None
            else np.load(adv_path, mmap_mode="r")
        )
        if (
            self.adv_iq is not None
            and self.adv_iq.shape != self.clean_iq.shape
        ):
            raise RuntimeError(
                "Offline attack shape mismatch: "
                f"clean={self.clean_iq.shape}, adv={self.adv_iq.shape}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        clean = torch.from_numpy(self.clean_iq[idx])
        if self.adv_iq is None:
            adv = clean
        else:
            # mmap is read-only; make a writable slice for torch.
            adv = torch.from_numpy(
                np.array(self.adv_iq[idx], dtype=np.float32, copy=True)
            )
        return clean, adv, torch.tensor(self.labels[idx], dtype=torch.long)


# =============================================================================
# Attack statistics
# =============================================================================


@dataclass
class AttackStats:
    attack: str
    epsilon: float
    total: int = 0
    clean_correct: int = 0
    adv_correct: int = 0
    success: int = 0
    ce_sum: float = 0.0
    linf_sum: float = 0.0
    l2_sum: float = 0.0
    power_sum: float = 0.0
    psr_sum: float = 0.0
    ober_sum: float = 0.0

    def update(
        self,
        clean_pred: torch.Tensor,
        adv_logits: torch.Tensor,
        labels: torch.Tensor,
        metric_tuple,
    ) -> None:
        adv_pred = adv_logits.argmax(dim=1)
        clean_correct = clean_pred.eq(labels)
        success = clean_correct & adv_pred.ne(labels)
        n = int(len(labels))

        self.total += n
        self.clean_correct += int(clean_correct.sum().item())
        self.adv_correct += int(adv_pred.eq(labels).sum().item())
        self.success += int(success.sum().item())
        self.ce_sum += float(
            F.cross_entropy(adv_logits, labels, reduction="sum").item()
        )

        linf, l2, power, psr, ober = metric_tuple
        self.linf_sum += float(linf.sum().item())
        self.l2_sum += float(l2.sum().item())
        self.power_sum += float(power.sum().item())
        self.psr_sum += float(psr.sum().item())
        self.ober_sum += float(ober.sum().item())

    def row(self) -> Dict[str, Any]:
        n = max(self.total, 1)
        cc = max(self.clean_correct, 1)
        return {
            "Attack": self.attack,
            "EPS": self.epsilon,
            "Samples": self.total,
            "Source_Clean_Acc": self.clean_correct / n,
            "Source_Adv_Acc": self.adv_correct / n,
            "Source_ASR_on_CleanCorrect": self.success / cc,
            "Mean_Adv_CE": self.ce_sum / n,
            "Mean_Linf": self.linf_sum / n,
            "Mean_L2": self.l2_sum / n,
            "Mean_Power": self.power_sum / n,
            "Mean_PSR_dB": self.psr_sum / n,
            "Mean_OBER": self.ober_sum / n,
        }


def score_cached_corpus(
    attack_name: str,
    path: Path,
    loader: DataLoader,
    evaluation_model: nn.Module,
    evaluation_norm: common.GlobalRMSNormalizer,
    evaluation_model_name: str,
    evaluator,
    epsilon: float,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    arr = np.load(path, mmap_mode="r")
    global_stats = AttackStats(attack_name, epsilon)
    per_snr: Dict[float, AttackStats] = {}

    for physical, labels, snrs, indices in tqdm(
        loader,
        desc=f"Score {attack_name} on {evaluation_model_name}",
        leave=False,
    ):
        physical = physical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        snrs = snrs.to(device, non_blocking=True)
        adv_phys = torch.from_numpy(
            np.array(arr[indices.numpy()], dtype=np.float32, copy=True)
        ).to(device, non_blocking=True)

        with torch.no_grad():
            clean_norm = normalize_physical(physical, evaluation_norm)
            adv_norm = normalize_physical(adv_phys, evaluation_norm)
            clean_pred = evaluation_model(clean_norm).argmax(dim=1)
            adv_logits = evaluation_model(adv_norm)
            metrics = evaluator.perturbation_stats_per_sample(
                clean_norm,
                adv_norm,
                keep_ratio=0.25,
            )
        global_stats.update(clean_pred, adv_logits, labels, metrics)

        for s in torch.unique(snrs):
            sv = float(s.item())
            mask = torch.isclose(snrs, s)
            if sv not in per_snr:
                per_snr[sv] = AttackStats(attack_name, epsilon)
            with torch.no_grad():
                cn = normalize_physical(physical[mask], evaluation_norm)
                an = normalize_physical(adv_phys[mask], evaluation_norm)
                cp = evaluation_model(cn).argmax(dim=1)
                al = evaluation_model(an)
                mt = evaluator.perturbation_stats_per_sample(
                    cn, an, keep_ratio=0.25
                )
            per_snr[sv].update(cp, al, labels[mask], mt)

        del physical, labels, snrs, adv_phys

    row = global_stats.row()
    row["Evaluation_Model"] = evaluation_model_name

    snr_rows: List[Dict[str, Any]] = []
    for snr, stats in sorted(per_snr.items()):
        r = stats.row()
        r["SNR_dB"] = snr
        r["Evaluation_Model"] = evaluation_model_name
        snr_rows.append(r)

    return row, snr_rows


# =============================================================================
# Diffusion attack generation with AD-2-consistent tanh temperature
# =============================================================================


@torch.no_grad()
def generate_diffusion_candidate_bank_consistent(
    shared,
    attack_diffusion: nn.Module,
    attack_normalizer: common.GlobalRMSNormalizer,
    attack_betas: torch.Tensor,
    physical_clean: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    num_starts: int,
    tanh_temperature: float,
    attack_sampler: str,
    attack_sampling_steps: int,
    attack_ddim_eta: float,
) -> torch.Tensor:
    if num_starts < 1:
        raise ValueError("num_starts must be >= 1")
    if tanh_temperature <= 0:
        raise ValueError("tanh_temperature must be positive")

    batch = len(physical_clean)
    if batch == 0:
        return physical_clean.unsqueeze(1).expand(
            -1, num_starts, -1, -1
        )

    clean_attack = shared.normalize_torch(
        physical_clean, attack_normalizer
    )
    clean_multi = clean_attack.repeat_interleave(num_starts, dim=0)
    labels_multi = labels.repeat_interleave(num_starts, dim=0)

    if attack_sampler == "ddpm_full":
        raw_delta = shared.generate_delta_full_ddpm(
            model=attack_diffusion,
            clean=clean_multi,
            labels=labels_multi,
            betas=attack_betas,
        )
    elif attack_sampler == "ddim":
        raw_delta = shared.generate_delta_fast_ddim(
            model=attack_diffusion,
            clean=clean_multi,
            labels=labels_multi,
            betas=attack_betas,
            sampling_steps=int(attack_sampling_steps),
            eta=float(attack_ddim_eta),
        )
    else:
        raise ValueError(f"Unknown attack_sampler={attack_sampler!r}")

    unit = torch.tanh(
        raw_delta.float() / float(tanh_temperature)
    )
    delta = float(eps) * unit
    adv_attack = torch.clamp(
        clean_multi + delta,
        -common.INPUT_CLIP,
        common.INPUT_CLIP,
    )
    adv_physical = shared.denormalize_torch(
        adv_attack,
        attack_normalizer,
    )
    return adv_physical.view(
        batch,
        num_starts,
        *physical_clean.shape[1:],
    ).detach()


@torch.no_grad()
def precompute_diffusion_unit_cache_consistent(
    shared,
    loader: DataLoader,
    attack_diffusion: nn.Module,
    attack_normalizer: common.GlobalRMSNormalizer,
    attack_betas: torch.Tensor,
    tanh_temperature: float,
    num_starts: int,
    device: torch.device,
    seed: int,
):
    """Cache unit=tanh(raw/T) once so the final epsilon sweep is paired."""
    common.set_global_seed(int(seed))
    cache = []

    for physical, labels, _snrs in tqdm(
        loader,
        desc="Precompute Target-guided paired DDPM bank",
    ):
        physical = physical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        clean_attack = shared.normalize_torch(
            physical,
            attack_normalizer,
        )
        clean_multi = clean_attack.repeat_interleave(
            num_starts,
            dim=0,
        )
        labels_multi = labels.repeat_interleave(
            num_starts,
            dim=0,
        )

        raw_delta = shared.generate_delta_full_ddpm(
            model=attack_diffusion,
            clean=clean_multi,
            labels=labels_multi,
            betas=attack_betas,
        )
        unit = torch.tanh(
            raw_delta.float() / float(tanh_temperature)
        ).view(
            len(physical),
            num_starts,
            *physical.shape[1:],
        ).detach().cpu()

        if torch.cuda.is_available():
            unit = unit.pin_memory()
        cache.append(unit)

        del physical, labels, clean_attack, clean_multi, labels_multi
        del raw_delta, unit

    clean_cuda()
    return cache


# =============================================================================
# Fixed offline corpus generation
# =============================================================================


ATTACK_DISPLAY = {
    "diffusion": "Target-Diffusion",
    "pgd": "Target-PGD",
}

TRAIN_DISPLAY = {
    "clean": "Clean-Finetune-Surrogate",
    "diffusion": "Diffusion-Offline-Surrogate",
    "pgd": "PGD-Offline-Surrogate",
}

TRAIN_MODE_ORDER = ["clean", "diffusion", "pgd"]
ATTACK_MODE_ORDER = ["diffusion", "pgd"]


def build_cache_paths(
    root: Path,
    eps: float,
    pgd_steps: int,
    diffusion_k: int,
    n: int,
    seed: int,
    target_hash: str,
    diffusion_hash: str,
) -> Dict[str, Path]:
    tag = eps_tag(eps)
    return {
        "diffusion": root / (
            f"target_diffusionK{diffusion_k}_eps{tag}_n{n}_seed{seed}_"
            f"T{target_hash[:8]}_D{diffusion_hash[:8]}.npy"
        ),
        "pgd": root / (
            f"target_pgd{pgd_steps}_eps{tag}_n{n}_seed{seed}_"
            f"T{target_hash[:8]}.npy"
        ),
    }


def valid_npy(path: Path, expected_shape: Tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    try:
        x = np.load(path, mmap_mode="r")
        return (
            x.dtype == np.float32
            and tuple(x.shape) == tuple(expected_shape)
        )
    except Exception:
        return False


def generate_fixed_corpus(
    mode: str,
    out_path: Path,
    loader: DataLoader,
    source_shape: Tuple[int, ...],
    source_target: nn.Module,
    target_norm: common.GlobalRMSNormalizer,
    shared,
    diffusion: nn.Module,
    diffusion_norm: common.GlobalRMSNormalizer,
    diffusion_betas: torch.Tensor,
    diffusion_temperature: float,
    epsilon: float,
    pgd_steps: int,
    diffusion_k: int,
    diffusion_sampler: str,
    diffusion_sampling_steps: int,
    diffusion_ddim_eta: float,
    device: torch.device,
    seed: int,
) -> None:
    """Generate one target-source fixed physical-IQ adversarial corpus."""
    if mode not in ATTACK_MODE_ORDER:
        raise ValueError(mode)

    common.set_global_seed(seed)
    mmap = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=source_shape,
    )

    description = (
        f"Generate fixed TargetPGD{pgd_steps}"
        if mode == "pgd"
        else f"Generate fixed TargetDiffusionK{diffusion_k}"
    )

    for physical, labels, _snrs, indices in tqdm(
        loader,
        desc=description,
    ):
        physical = physical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mode == "pgd":
            # White-box only with respect to the FIXED source TargetAMC.
            # The defense SurrogateAMC is never consulted here.
            with torch.enable_grad():
                adv_phys = shared.gradient_attack_physical(
                    attack_name="PGD",
                    model=source_target,
                    model_normalizer=target_norm,
                    physical_clean=physical,
                    labels=labels,
                    eps=float(epsilon),
                    steps=int(pgd_steps),
                )

        else:  # diffusion
            # K=1 needs no AMC query at corpus-generation time.
            with torch.no_grad():
                bank = generate_diffusion_candidate_bank_consistent(
                    shared=shared,
                    attack_diffusion=diffusion,
                    attack_normalizer=diffusion_norm,
                    attack_betas=diffusion_betas,
                    physical_clean=physical,
                    labels=labels,
                    eps=float(epsilon),
                    num_starts=int(diffusion_k),
                    tanh_temperature=float(diffusion_temperature),
                    attack_sampler=diffusion_sampler,
                    attack_sampling_steps=int(diffusion_sampling_steps),
                    attack_ddim_eta=float(diffusion_ddim_eta),
                )

            if int(diffusion_k) == 1:
                adv_phys = bank[:, 0].detach()
            else:
                # Optional stronger ablation: both methods have the same source
                # TargetAMC knowledge. Selection uses only TargetAMC CE and no
                # gradient refinement. Main experiment should keep K=1.
                adv_phys, _ = shared.select_hardest_from_candidate_bank(
                    candidate_physical=bank,
                    physical_clean=physical,
                    labels=labels,
                    robust_amc=source_target,
                    amc_normalizer=target_norm,
                    eps=float(epsilon),
                    refinement_steps=0,
                    refinement_step_size=None,
                )
            del bank

        idx = indices.numpy()
        mmap[idx] = (
            adv_phys.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        del physical, labels, adv_phys

    mmap.flush()
    del mmap
    clean_cuda()


# =============================================================================
# Fixed validation attack caches (NO current-defense gradients)
# =============================================================================


@torch.no_grad()
def cache_validation_diffusion(
    shared,
    loader: DataLoader,
    diffusion: nn.Module,
    diffusion_norm: common.GlobalRMSNormalizer,
    betas: torch.Tensor,
    tanh_temperature: float,
    epsilon: float,
    device: torch.device,
    seed: int,
):
    common.set_global_seed(seed)
    cache = []
    for physical, labels, _ in tqdm(
        loader,
        desc="Cache validation Target-Diffusion",
        leave=False,
    ):
        physical = physical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bank = generate_diffusion_candidate_bank_consistent(
            shared=shared,
            attack_diffusion=diffusion,
            attack_normalizer=diffusion_norm,
            attack_betas=betas,
            physical_clean=physical,
            labels=labels,
            eps=float(epsilon),
            num_starts=1,
            tanh_temperature=float(tanh_temperature),
            attack_sampler="ddpm_full",
            attack_sampling_steps=len(betas),
            attack_ddim_eta=0.0,
        )
        cache.append(bank[:, 0].detach().cpu())
        del physical, labels, bank
    clean_cuda()
    return cache


def cache_validation_target_pgd(
    shared,
    loader: DataLoader,
    source_target: nn.Module,
    target_norm: common.GlobalRMSNormalizer,
    epsilon: float,
    pgd_steps: int,
    device: torch.device,
    seed: int,
):
    """Craft PGD once on fixed TargetAMC; never use moving Surrogate gradients."""
    common.set_global_seed(seed)
    cache = []
    for physical, labels, _ in tqdm(
        loader,
        desc="Cache validation Target-PGD",
        leave=False,
    ):
        physical = physical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.enable_grad():
            adv_phys = shared.gradient_attack_physical(
                attack_name="PGD",
                model=source_target,
                model_normalizer=target_norm,
                physical_clean=physical,
                labels=labels,
                eps=float(epsilon),
                steps=int(pgd_steps),
            )
        cache.append(adv_phys.detach().cpu())
        del physical, labels, adv_phys
    clean_cuda()
    return cache


def validate_surrogate_fixed_attacks(
    model: nn.Module,
    surrogate_norm: common.GlobalRMSNormalizer,
    loader: DataLoader,
    fixed_diff_cache,
    fixed_pgd_cache,
    shared,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validation uses only fixed target-source attacks.

    No gradients are computed through `model`; therefore checkpoint selection
    does not adapt any attack to the moving defense model.  In addition to the
    original accuracy/robustness metrics, this routine records cross-entropy on
    the clean, fixed Target-PGD and fixed Target-Diffusion validation sets.
    All validation losses are averaged over ALL validation samples.
    """
    was_training = model.training
    model.eval()

    total = 0
    clean_correct_n = 0
    pgd_robust_n = 0
    diff_robust_n = 0
    clean_loss_sum = 0.0
    pgd_loss_sum = 0.0
    diff_loss_sum = 0.0

    with torch.no_grad():
        for bi, (physical, labels, _snrs) in enumerate(loader):
            physical = physical.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            clean = shared.normalize_torch(physical, surrogate_norm)
            clean_logits = model(clean)
            clean_pred = clean_logits.argmax(dim=1)
            cc = clean_pred.eq(labels)

            diff_phys = fixed_diff_cache[bi].to(device, non_blocking=True)
            pgd_phys = fixed_pgd_cache[bi].to(device, non_blocking=True)
            diff_norm = shared.normalize_torch(diff_phys, surrogate_norm)
            pgd_norm = shared.normalize_torch(pgd_phys, surrogate_norm)
            diff_logits = model(diff_norm)
            pgd_logits = model(pgd_norm)
            diff_pred = diff_logits.argmax(dim=1)
            pgd_pred = pgd_logits.argmax(dim=1)

            batch_n = int(len(labels))
            total += batch_n
            clean_correct_n += int(cc.sum().item())
            diff_robust_n += int((cc & diff_pred.eq(labels)).sum().item())
            pgd_robust_n += int((cc & pgd_pred.eq(labels)).sum().item())

            clean_loss_sum += float(
                F.cross_entropy(clean_logits, labels, reduction="sum").item()
            )
            pgd_loss_sum += float(
                F.cross_entropy(pgd_logits, labels, reduction="sum").item()
            )
            diff_loss_sum += float(
                F.cross_entropy(diff_logits, labels, reduction="sum").item()
            )

    if was_training:
        model.train()

    denom = max(clean_correct_n, 1)
    sample_denom = max(total, 1)
    return {
        "Clean_Acc": clean_correct_n / sample_denom,
        "Val_Clean_Loss": clean_loss_sum / sample_denom,
        "Val_TargetPGD_Loss": pgd_loss_sum / sample_denom,
        "Val_TargetDiffusion_Loss": diff_loss_sum / sample_denom,
        "Fixed_TargetPGD_Robust_Acc_CC": pgd_robust_n / denom,
        "Fixed_TargetPGD_ASR_CC": 1.0 - pgd_robust_n / denom,
        "Fixed_TargetDiffusion_Robust_Acc_CC": diff_robust_n / denom,
        "Fixed_TargetDiffusion_ASR_CC": 1.0 - diff_robust_n / denom,
    }


# =============================================================================
# Offline SurrogateAMC training
# =============================================================================


def train_surrogate_mode(
    mode: str,
    surrogate_ckpt: dict,
    surrogate_norm: common.GlobalRMSNormalizer,
    clean_iq: np.ndarray,
    labels: np.ndarray,
    adv_path: Optional[Path],
    val_loader: DataLoader,
    fixed_val_diff,
    fixed_val_pgd,
    shared,
    device: torch.device,
    output_path: Path,
    args,
) -> Tuple[nn.Module, dict, pd.DataFrame]:
    if mode not in TRAIN_MODE_ORDER:
        raise ValueError(mode)

    # Every defense starts from EXACTLY the same original SurrogateAMC weights.
    model = clone_surrogate(surrogate_ckpt, device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    dataset = OfflinePairDataset(
        clean_iq=clean_iq,
        labels=labels,
        adv_path=None if mode == "clean" else adv_path,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    weight_sum = (
        args.checkpoint_clean_weight
        + args.checkpoint_target_pgd_weight
        + args.checkpoint_diffusion_weight
    )
    wc = args.checkpoint_clean_weight / weight_sum
    wp = args.checkpoint_target_pgd_weight / weight_sum
    wd = args.checkpoint_diffusion_weight / weight_sum

    best_score = -float("inf")
    best_state = None
    best_epoch = None
    best_val = None
    history: List[Dict[str, Any]] = []

    print("\n" + "=" * 112)
    print(f" OFFLINE SURROGATE HARDENING: {TRAIN_DISPLAY[mode]}")
    print("=" * 112)
    print(
        "[INFO] Training uses only clean/fixed stored samples. "
        "No attack is generated against the moving SurrogateAMC."
    )
    print(
        "[INFO] Checkpoint selection uses only fixed Target-PGD / "
        "fixed Target-Diffusion validation attacks."
    )
    print(
        "[INFO] Early stopping is disabled: all epochs are trained and logged; "
        "the best checkpoint is selected after the full run."
    )

    for epoch in range(1, args.epochs + 1):
        # Same shuffle order in every mode/epoch.
        generator = torch.Generator().manual_seed(
            args.seed + 404 + epoch
        )
        loader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(
                args.persistent_workers and args.num_workers > 0
            ),
            generator=generator,
            drop_last=False,
        )

        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        adv_seen = 0

        for clean_phys, adv_phys, y in tqdm(
            loader,
            desc=f"{TRAIN_DISPLAY[mode]} epoch {epoch}",
        ):
            clean_phys = clean_phys.to(device, non_blocking=True)
            adv_phys = adv_phys.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # IMPORTANT: defense always uses SURROGATE normalizer.
            clean_norm = shared.normalize_torch(
                clean_phys,
                surrogate_norm,
            )

            if mode == "clean":
                mixed = clean_norm
                num_adv = 0
            else:
                adv_norm = shared.normalize_torch(
                    adv_phys,
                    surrogate_norm,
                )
                num_adv = int(len(y) * args.adv_ratio)
                if args.adv_ratio > 0 and len(y):
                    num_adv = max(1, num_adv)
                mixed = clean_norm.clone()
                if num_adv:
                    mixed[:num_adv] = adv_norm[:num_adv]
                adv_seen += num_adv

            optimizer.zero_grad(set_to_none=True)
            logits = model(mixed)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += float(loss.detach().item()) * len(y)
            correct += int(logits.argmax(dim=1).eq(y).sum().item())
            total += int(len(y))

        val = validate_surrogate_fixed_attacks(
            model=model,
            surrogate_norm=surrogate_norm,
            loader=val_loader,
            fixed_diff_cache=fixed_val_diff,
            fixed_pgd_cache=fixed_val_pgd,
            shared=shared,
            device=device,
        )

        score = (
            wc * val["Clean_Acc"]
            + wp * val["Fixed_TargetPGD_Robust_Acc_CC"]
            + wd * val["Fixed_TargetDiffusion_Robust_Acc_CC"]
        )
        val_loss = (
            wc * val["Val_Clean_Loss"]
            + wp * val["Val_TargetPGD_Loss"]
            + wd * val["Val_TargetDiffusion_Loss"]
        )

        row = {
            "Mode": TRAIN_DISPLAY[mode],
            "Epoch": epoch,
            "Train_Loss": loss_sum / max(total, 1),
            "Train_Acc": correct / max(total, 1),
            "Val_Loss": val_loss,
            "Adversarial_Samples_Seen": adv_seen,
            **val,
            "Checkpoint_Score": score,
        }
        history.append(row)

        print(
            f"[{TRAIN_DISPLAY[mode]}] epoch={epoch:02d} "
            f"trainLoss={row['Train_Loss']:.4f} "
            f"cleanValLoss={val['Val_Clean_Loss']:.4f} "
            f"train={100*row['Train_Acc']:.2f}% | "
            f"clean={100*val['Clean_Acc']:.2f}% "
            f"fixedTargetPGDrob="
            f"{100*val['Fixed_TargetPGD_Robust_Acc_CC']:.2f}% "
            f"fixedDiffrob="
            f"{100*val['Fixed_TargetDiffusion_Robust_Acc_CC']:.2f}% "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val = {**val, "Val_Loss": val_loss}
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            print(f"  -> BEST {TRAIN_DISPLAY[mode]} epoch={epoch}")

        clean_cuda()

    if best_state is None:
        raise RuntimeError(f"No best checkpoint for mode={mode}")

    for r in history:
        r["Selected_Checkpoint"] = bool(int(r["Epoch"]) == int(best_epoch))

    model.load_state_dict(best_state, strict=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state": model.cpu().state_dict(),
        "normalizer": surrogate_norm.state_dict(),
        "training": "target_source_fixed_offline_hardening_surrogate_amc",
        "offline_mode": mode,
        "display_name": TRAIN_DISPLAY[mode],
        "defense_model_role": "surrogate",
        "defense_model_class": "SurrogateAMC",
        "attack_source_model_role": (
            "none_clean_control" if mode == "clean" else "original_target"
        ),
        "attacks_generated_against_moving_defense": False,
        "checkpoint_selection_uses_current_defense_gradients": False,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "best_validation": best_val,
        "full_epoch_training": True,
        "configured_epochs": int(args.epochs),
        "completed_epochs": len(history),
        "early_stopping_enabled": False,
        "cross_method_loss_metric": "Val_Clean_Loss",
        "train_epsilon_in_target_coordinate": args.train_eps,
        "adv_ratio": 0.0 if mode == "clean" else args.adv_ratio,
        "source_attack_file": (
            None if adv_path is None else str(adv_path)
        ),
        "history": history,
    }
    common.save_artifact(ckpt, output_path)

    eval_model = common.SurrogateAMC().to(device)
    eval_model.load_state_dict(best_state, strict=True)
    freeze_model(eval_model)
    return eval_model, ckpt, pd.DataFrame(history)



# =============================================================================
# Diversity-aware offline banks / resumable experiment orchestration
# =============================================================================


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    display_name: str
    attack_bank_keys: Tuple[str, ...]
    variant_policy: str
    role: str
    checkpoint_filename: str
    legacy_mode: Optional[str] = None


class DiverseOfflineDataset(Dataset):
    """Clean physical IQ paired with one or more fixed adversarial banks.

    Every bank may be either:
      [N, 2, L]       : one adversarial variant per clean sample
      [N, V, 2, L]    : V adversarial variants per clean sample

    variant_policy="cycle"
        Dataset length remains N.  At epoch e each clean sample deterministically
        rotates through the global variant bank.  This keeps the number of
        optimizer updates comparable to PGD while exposing the model to many
        distinct attacks across epochs.
    """

    def __init__(
        self,
        clean_iq: np.ndarray,
        labels: np.ndarray,
        adv_paths: Sequence[Path],
        variant_policy: str,
        seed: int,
    ):
        self.clean_iq = clean_iq
        self.labels = labels
        self.variant_policy = str(variant_policy)
        self.seed = int(seed)
        self.epoch = 1

        if self.variant_policy not in {"clean", "cycle"}:
            raise ValueError(f"Unknown variant_policy={variant_policy!r}")

        self.banks = []
        self.variant_map: List[Tuple[int, int]] = []
        expected_tail = tuple(clean_iq.shape[1:])
        n = len(clean_iq)

        for bank_index, path in enumerate(adv_paths):
            arr = np.load(path, mmap_mode="r")
            if arr.dtype != np.float32:
                raise RuntimeError(f"Attack bank must be float32: {path}")
            if arr.ndim == 3:
                if arr.shape[0] != n or tuple(arr.shape[1:]) != expected_tail:
                    raise RuntimeError(
                        f"Attack bank shape mismatch for {path}: "
                        f"bank={arr.shape}, clean={clean_iq.shape}"
                    )
                variants = 1
            elif arr.ndim == 4:
                if arr.shape[0] != n or tuple(arr.shape[2:]) != expected_tail:
                    raise RuntimeError(
                        f"Attack bank shape mismatch for {path}: "
                        f"bank={arr.shape}, clean={clean_iq.shape}"
                    )
                variants = int(arr.shape[1])
            else:
                raise RuntimeError(
                    f"Attack bank must be [N,C,L] or [N,V,C,L], got {arr.shape}: {path}"
                )

            self.banks.append(arr)
            for variant_index in range(variants):
                self.variant_map.append((bank_index, variant_index))

        if self.variant_policy == "clean":
            self.variant_map = []
        elif not self.variant_map:
            raise RuntimeError("Adversarial experiment has no attack variants.")

    @property
    def num_variants(self) -> int:
        return len(self.variant_map)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.labels)

    def _resolve_index(self, idx: int) -> Tuple[int, Optional[Tuple[int, int]]]:
        n = len(self.labels)
        if self.variant_policy == "clean":
            return idx, None

        clean_index = idx
        # Per-sample phase offset prevents one mini-batch from using only one
        # variant while still guaranteeing a complete cycle over V epochs.
        global_variant = (
            (clean_index * 9973 + self.seed) + (self.epoch - 1)
        ) % self.num_variants

        return clean_index, self.variant_map[global_variant]

    def __getitem__(self, idx):
        clean_index, variant_ref = self._resolve_index(int(idx))
        clean = torch.from_numpy(self.clean_iq[clean_index])

        if variant_ref is None:
            adv = clean
        else:
            bank_index, variant_index = variant_ref
            bank = self.banks[bank_index]
            if bank.ndim == 3:
                adv_np = bank[clean_index]
            else:
                adv_np = bank[clean_index, variant_index]
            adv = torch.from_numpy(
                np.array(adv_np, dtype=np.float32, copy=True)
            )

        return (
            clean,
            adv,
            torch.tensor(self.labels[clean_index], dtype=torch.long),
        )


def valid_attack_bank(
    path: Path,
    n: int,
    tail_shape: Tuple[int, ...],
    variants: int,
) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        if variants == 1:
            valid_shapes = {
                (n, *tail_shape),
                (n, 1, *tail_shape),
            }
            return arr.dtype == np.float32 and tuple(arr.shape) in valid_shapes
        return (
            arr.dtype == np.float32
            and tuple(arr.shape) == (n, variants, *tail_shape)
        )
    except Exception:
        return False


def diversity_cache_paths(
    root: Path,
    eps: float,
    multistart_k: int,
    ddim_steps: Sequence[int],
    n: int,
    seed: int,
    target_hash: str,
    diffusion_hash: str,
) -> Dict[str, Path]:
    tag = eps_tag(eps)
    step_tag = "-".join(str(int(x)) for x in ddim_steps)
    return {
        "multistart": root / (
            f"target_diffusion_multistartK{multistart_k}_eps{tag}_n{n}_seed{seed}_"
            f"T{target_hash[:8]}_D{diffusion_hash[:8]}.npy"
        ),
        "multistep": root / (
            f"target_diffusion_ddimsteps{step_tag}_eps{tag}_n{n}_seed{seed}_"
            f"T{target_hash[:8]}_D{diffusion_hash[:8]}.npy"
        ),
    }


@torch.no_grad()
def generate_multistart_diffusion_bank(
    out_path: Path,
    loader: DataLoader,
    source_shape: Tuple[int, ...],
    shared,
    diffusion: nn.Module,
    diffusion_norm: common.GlobalRMSNormalizer,
    diffusion_betas: torch.Tensor,
    diffusion_temperature: float,
    epsilon: float,
    multistart_k: int,
    device: torch.device,
    seed: int,
) -> None:
    """Store K independent full-DDPM samples for every clean source row."""
    if multistart_k < 2:
        raise ValueError("multistart_k must be >= 2 for a diversity bank")

    common.set_global_seed(int(seed))
    shape = (
        source_shape[0],
        int(multistart_k),
        *source_shape[1:],
    )
    mmap = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    for physical, labels, _snrs, indices in tqdm(
        loader,
        desc=f"Generate Diffusion multi-start K={multistart_k}",
    ):
        physical = physical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # Generate starts sequentially instead of repeat_interleave(K).
        # This keeps peak GPU memory close to the K=1 path on an 8 GB GPU.
        candidates = []
        for _start in range(int(multistart_k)):
            one = generate_diffusion_candidate_bank_consistent(
                shared=shared,
                attack_diffusion=diffusion,
                attack_normalizer=diffusion_norm,
                attack_betas=diffusion_betas,
                physical_clean=physical,
                labels=labels,
                eps=float(epsilon),
                num_starts=1,
                tanh_temperature=float(diffusion_temperature),
                attack_sampler="ddpm_full",
                attack_sampling_steps=len(diffusion_betas),
                attack_ddim_eta=0.0,
            )
            candidates.append(one[:, 0])
        bank = torch.stack(candidates, dim=1)
        mmap[indices.numpy()] = (
            bank.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        del physical, labels, candidates, bank

    mmap.flush()
    del mmap
    clean_cuda()


@torch.no_grad()
def generate_multistep_diffusion_bank(
    out_path: Path,
    loader: DataLoader,
    source_shape: Tuple[int, ...],
    shared,
    diffusion: nn.Module,
    diffusion_norm: common.GlobalRMSNormalizer,
    diffusion_betas: torch.Tensor,
    diffusion_temperature: float,
    epsilon: float,
    ddim_steps: Sequence[int],
    ddim_eta: float,
    device: torch.device,
    seed: int,
) -> None:
    """Store one DDIM sample for each requested reverse sampling depth.

    This changes the number of denoising network calls / skipped reverse steps;
    every variant still reaches the final x0 estimate.  It is therefore a
    sampling-depth diversity experiment, not an early-stopped noisy sample.
    """
    steps = [int(x) for x in ddim_steps]
    if not steps:
        raise ValueError("ddim_steps cannot be empty")

    common.set_global_seed(int(seed))
    shape = (
        source_shape[0],
        len(steps),
        *source_shape[1:],
    )
    mmap = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    for physical, labels, _snrs, indices in tqdm(
        loader,
        desc=f"Generate Diffusion DDIM-depth bank {steps}",
    ):
        physical = physical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        variants = []
        for sampling_steps in steps:
            candidate = generate_diffusion_candidate_bank_consistent(
                shared=shared,
                attack_diffusion=diffusion,
                attack_normalizer=diffusion_norm,
                attack_betas=diffusion_betas,
                physical_clean=physical,
                labels=labels,
                eps=float(epsilon),
                num_starts=1,
                tanh_temperature=float(diffusion_temperature),
                attack_sampler="ddim",
                attack_sampling_steps=int(sampling_steps),
                attack_ddim_eta=float(ddim_eta),
            )
            variants.append(candidate[:, 0])
        bank = torch.stack(variants, dim=1)
        mmap[indices.numpy()] = (
            bank.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        del physical, labels, variants, bank

    mmap.flush()
    del mmap
    clean_cuda()


def score_cached_attack_bank(
    attack_name: str,
    path: Path,
    loader: DataLoader,
    evaluation_model: nn.Module,
    evaluation_norm: common.GlobalRMSNormalizer,
    evaluation_model_name: str,
    evaluator,
    epsilon: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Aggregate attack quality over every stored variant in a bank."""
    arr = np.load(path, mmap_mode="r")
    variants = 1 if arr.ndim == 3 else int(arr.shape[1])
    stats = AttackStats(attack_name, epsilon)

    for physical, labels, _snrs, indices in tqdm(
        loader,
        desc=f"Score {attack_name} on {evaluation_model_name}",
        leave=False,
    ):
        physical = physical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        raw = np.array(arr[indices.numpy()], dtype=np.float32, copy=True)
        if arr.ndim == 3:
            raw = raw[:, None]
        adv_phys = torch.from_numpy(raw).to(device, non_blocking=True)
        batch = len(labels)

        clean_norm = normalize_physical(physical, evaluation_norm)
        clean_rep = clean_norm.repeat_interleave(variants, dim=0)
        labels_rep = labels.repeat_interleave(variants, dim=0)
        adv_flat = adv_phys.reshape(
            batch * variants,
            *physical.shape[1:],
        )
        adv_norm = normalize_physical(adv_flat, evaluation_norm)

        with torch.no_grad():
            clean_pred = evaluation_model(clean_norm).argmax(dim=1)
            clean_pred_rep = clean_pred.repeat_interleave(variants, dim=0)
            adv_logits = evaluation_model(adv_norm)
            metrics = evaluator.perturbation_stats_per_sample(
                clean_rep,
                adv_norm,
                keep_ratio=0.25,
            )
        stats.update(clean_pred_rep, adv_logits, labels_rep, metrics)
        del physical, labels, adv_phys, adv_flat, adv_norm

    row = stats.row()
    row["Stored_Variants_Per_Clean"] = variants
    row["Evaluation_Model"] = evaluation_model_name
    return row


def compute_attack_bank_diversity(
    bank_name: str,
    path: Path,
    clean_iq: np.ndarray,
    normalizer: common.GlobalRMSNormalizer,
    max_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Measure whether a stored attack bank is genuinely diverse."""
    arr = np.load(path, mmap_mode="r")
    variants = 1 if arr.ndim == 3 else int(arr.shape[1])
    n = len(clean_iq)
    take = min(n, int(max_samples)) if max_samples > 0 else n
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(n, size=take, replace=False)) if take < n else np.arange(n)

    if arr.ndim == 3:
        adv = np.asarray(arr[indices], dtype=np.float32)[:, None]
    else:
        adv = np.asarray(arr[indices], dtype=np.float32)
    clean = np.asarray(clean_iq[indices], dtype=np.float32)[:, None]
    scale = float(normalizer.scale)
    delta = (adv - clean) / scale
    flat = delta.reshape(take, variants, -1).astype(np.float64, copy=False)

    if variants < 2:
        pair_cos = float("nan")
        pair_l2 = float("nan")
        coord_std = 0.0
    else:
        norms = np.linalg.norm(flat, axis=2, keepdims=True)
        unit = flat / np.maximum(norms, 1e-12)
        cos = np.einsum("nvd,nwd->nvw", unit, unit)
        tri = np.triu_indices(variants, k=1)
        pair_cos = float(cos[:, tri[0], tri[1]].mean())

        pair_values = []
        for i, j in zip(tri[0], tri[1]):
            pair_values.append(
                np.linalg.norm(flat[:, i] - flat[:, j], axis=1)
            )
        pair_l2 = float(np.stack(pair_values, axis=1).mean())
        coord_std = float(flat.std(axis=1).mean())

    return {
        "Bank": bank_name,
        "Path": str(path),
        "Samples_Used": int(take),
        "Variants_Per_Clean": int(variants),
        "Mean_Pairwise_Cosine": pair_cos,
        "Mean_Pairwise_L2_TargetCoord": pair_l2,
        "Mean_Coordinate_STD_TargetCoord": coord_std,
    }


def build_experiment_specs(
    seed: int,
    multistart_k: int,
    ddim_steps: Sequence[int],
) -> Dict[str, ExperimentSpec]:
    step_tag = "-".join(str(int(x)) for x in ddim_steps)
    return {
        "clean": ExperimentSpec(
            key="clean",
            display_name="Clean-Finetune-Surrogate",
            attack_bank_keys=(),
            variant_policy="clean",
            role="Clean fine-tuning control",
            checkpoint_filename=f"surrogate_amc_clean_finetune_seed{seed}.pt",
            legacy_mode="clean",
        ),
        "pgd": ExperimentSpec(
            key="pgd",
            display_name="PGD-Offline-Surrogate",
            attack_bank_keys=("pgd",),
            variant_policy="cycle",
            role="Same-source Target-PGD baseline",
            checkpoint_filename=f"surrogate_amc_targetpgd_offline_seed{seed}.pt",
            legacy_mode="pgd",
        ),
        "diffusion_k1": ExperimentSpec(
            key="diffusion_k1",
            display_name="Diffusion-K1-Offline-Surrogate",
            attack_bank_keys=("baseline_diffusion",),
            variant_policy="cycle",
            role="Original one-sample diffusion baseline",
            checkpoint_filename=f"surrogate_amc_targetdiffusion_offline_seed{seed}.pt",
            legacy_mode="diffusion",
        ),
        "diffusion_multistart_cycle": ExperimentSpec(
            key="diffusion_multistart_cycle",
            display_name=f"Diffusion-MultiStartK{multistart_k}-Cycle-Surrogate",
            attack_bank_keys=("multistart",),
            variant_policy="cycle",
            role="Budget-matched stochastic multi-start diversity",
            checkpoint_filename=(
                f"surrogate_amc_targetdiffusion_multistartK{multistart_k}_cycle_seed{seed}.pt"
            ),
        ),
        "diffusion_multistep_cycle": ExperimentSpec(
            key="diffusion_multistep_cycle",
            display_name=f"Diffusion-DDIMSteps{step_tag}-Cycle-Surrogate",
            attack_bank_keys=("multistep",),
            variant_policy="cycle",
            role="Budget-matched reverse-sampling-depth diversity",
            checkpoint_filename=(
                f"surrogate_amc_targetdiffusion_ddimsteps{step_tag}_cycle_seed{seed}.pt"
            ),
        ),
        "diffusion_diverse_cycle": ExperimentSpec(
            key="diffusion_diverse_cycle",
            display_name=(
                f"Diffusion-DiverseK{multistart_k}+Steps{step_tag}-Cycle-Surrogate"
            ),
            attack_bank_keys=("multistart", "multistep"),
            variant_policy="cycle",
            role="Budget-matched combined multi-start + sampling-depth diversity",
            checkpoint_filename=(
                f"surrogate_amc_targetdiffusion_diverseK{multistart_k}_steps{step_tag}_"
                f"cycle_seed{seed}.pt"
            ),
        ),
    }


def checkpoint_is_compatible(
    ckpt: dict,
    spec: ExperimentSpec,
    args,
    target_hash: str,
    surrogate_hash: str,
    diffusion_hash: str,
    diversity_signature: str,
) -> Tuple[bool, str]:
    if "model_state" not in ckpt:
        return False, "missing model_state"
    if int(ckpt.get("seed", -1)) != int(args.seed):
        return False, "seed mismatch"

    stored_eps = ckpt.get(
        "train_epsilon_in_target_coordinate",
        ckpt.get("train_epsilon", None),
    )
    if spec.key != "clean":
        if stored_eps is None or not np.isclose(float(stored_eps), float(args.train_eps)):
            return False, "train epsilon mismatch"
        stored_ratio = ckpt.get("adv_ratio", None)
        if stored_ratio is not None and not np.isclose(float(stored_ratio), float(args.adv_ratio)):
            return False, "adv_ratio mismatch"

    if ckpt.get("initial_surrogate_sha256") not in {None, surrogate_hash}:
        return False, "initial Surrogate checkpoint hash mismatch"
    if spec.key != "clean" and ckpt.get("source_target_sha256") not in {None, target_hash}:
        return False, "source Target checkpoint hash mismatch"

    if spec.key.startswith("diffusion"):
        if ckpt.get("diffusion_sha256") not in {None, diffusion_hash}:
            return False, "diffusion checkpoint hash mismatch"

    # New checkpoints carry an exact experiment signature.  Legacy checkpoints
    # from the immediately previous K=1/PGD/clean run are accepted by offline_mode.
    stored_key = ckpt.get("experiment_key")
    if stored_key is not None:
        if stored_key != spec.key:
            return False, f"experiment_key mismatch ({stored_key!r})"
        diversity_dependent = spec.key in {
            "diffusion_multistart_cycle",
            "diffusion_multistep_cycle",
            "diffusion_diverse_cycle",
        }
        if (
            diversity_dependent
            and ckpt.get("diversity_signature") != diversity_signature
        ):
            return False, "diversity signature mismatch"
    elif spec.legacy_mode is not None:
        if ckpt.get("offline_mode") != spec.legacy_mode:
            return False, "legacy offline_mode mismatch"
    else:
        return False, "new diversity experiment lacks experiment metadata"

    return True, "compatible"


def load_surrogate_checkpoint_model(
    path: Path,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    ckpt = common.load_artifact(path)
    model = common.SurrogateAMC().to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    freeze_model(model)
    return model, ckpt


def train_surrogate_experiment(
    spec: ExperimentSpec,
    surrogate_ckpt: dict,
    surrogate_norm: common.GlobalRMSNormalizer,
    clean_iq: np.ndarray,
    labels: np.ndarray,
    attack_bank_paths: Sequence[Path],
    val_loader: DataLoader,
    fixed_val_diff,
    fixed_val_pgd,
    shared,
    device: torch.device,
    output_path: Path,
    args,
    target_checkpoint_path: Path,
    target_hash: str,
    surrogate_checkpoint_path: Path,
    surrogate_hash: str,
    diffusion_path: Path,
    diffusion_hash: str,
    diversity_signature: str,
) -> Tuple[nn.Module, dict, pd.DataFrame]:
    model = clone_surrogate(surrogate_ckpt, device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    dataset = DiverseOfflineDataset(
        clean_iq=clean_iq,
        labels=labels,
        adv_paths=attack_bank_paths,
        variant_policy=spec.variant_policy,
        seed=args.seed + 77,
    )

    train_epochs = int(args.epochs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    weight_sum = (
        args.checkpoint_clean_weight
        + args.checkpoint_target_pgd_weight
        + args.checkpoint_diffusion_weight
    )
    wc = args.checkpoint_clean_weight / weight_sum
    wp = args.checkpoint_target_pgd_weight / weight_sum
    wd = args.checkpoint_diffusion_weight / weight_sum

    best_score = -float("inf")
    best_state = None
    best_epoch = None
    best_val = None
    history: List[Dict[str, Any]] = []

    print("\n" + "=" * 120)
    print(f" OFFLINE SURROGATE HARDENING: {spec.display_name}")
    print("=" * 120)
    print(
        f"[INFO] variant_policy={spec.variant_policy}, "
        f"stored_variants_per_clean={dataset.num_variants}, "
        f"dataset_pairs_per_epoch={len(dataset)}, epochs={train_epochs}"
    )
    print(
        "[INFO] No attack is generated against the moving SurrogateAMC; "
        "checkpoint selection uses fixed Target-source attacks only."
    )
    print(
        "[INFO] Early stopping is disabled: all epochs are trained and logged; "
        "the best checkpoint is selected after the full run."
    )

    for epoch in range(1, train_epochs + 1):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed + 404 + epoch)
        loader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(
                args.persistent_workers and args.num_workers > 0
            ),
            generator=generator,
            drop_last=False,
        )

        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        adv_seen = 0

        for clean_phys, adv_phys, y in tqdm(
            loader,
            desc=f"{spec.display_name} epoch {epoch}",
        ):
            clean_phys = clean_phys.to(device, non_blocking=True)
            adv_phys = adv_phys.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            clean_norm = shared.normalize_torch(clean_phys, surrogate_norm)
            if spec.variant_policy == "clean":
                mixed = clean_norm
                num_adv = 0
            else:
                adv_norm = shared.normalize_torch(adv_phys, surrogate_norm)
                num_adv = int(len(y) * args.adv_ratio)
                if args.adv_ratio > 0 and len(y):
                    num_adv = max(1, num_adv)
                mixed = clean_norm.clone()
                if num_adv:
                    mixed[:num_adv] = adv_norm[:num_adv]
                adv_seen += num_adv

            optimizer.zero_grad(set_to_none=True)
            logits = model(mixed)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += float(loss.detach().item()) * len(y)
            correct += int(logits.argmax(dim=1).eq(y).sum().item())
            total += int(len(y))

        val = validate_surrogate_fixed_attacks(
            model=model,
            surrogate_norm=surrogate_norm,
            loader=val_loader,
            fixed_diff_cache=fixed_val_diff,
            fixed_pgd_cache=fixed_val_pgd,
            shared=shared,
            device=device,
        )
        score = (
            wc * val["Clean_Acc"]
            + wp * val["Fixed_TargetPGD_Robust_Acc_CC"]
            + wd * val["Fixed_TargetDiffusion_Robust_Acc_CC"]
        )
        # Validation loss follows the SAME clean/Target-PGD/Target-Diffusion
        # weighting used by checkpoint selection.  The checkpoint itself is still
        # selected by the original robustness score so the experiment semantics
        # remain unchanged.
        val_loss = (
            wc * val["Val_Clean_Loss"]
            + wp * val["Val_TargetPGD_Loss"]
            + wd * val["Val_TargetDiffusion_Loss"]
        )
        row = {
            "Experiment_Key": spec.key,
            "Mode": spec.display_name,
            "Epoch": epoch,
            "Stored_Variants_Per_Clean": dataset.num_variants,
            "Variant_Policy": spec.variant_policy,
            "Pairs_Per_Epoch": len(dataset),
            "Train_Loss": loss_sum / max(total, 1),
            "Train_Acc": correct / max(total, 1),
            "Val_Loss": val_loss,
            "Adversarial_Samples_Seen": adv_seen,
            **val,
            "Checkpoint_Score": score,
        }
        history.append(row)
        print(
            f"[{spec.display_name}] epoch={epoch:02d} "
            f"trainLoss={row['Train_Loss']:.4f} "
            f"cleanValLoss={val['Val_Clean_Loss']:.4f} "
            f"mixedValLoss={row['Val_Loss']:.4f} "
            f"train={100*row['Train_Acc']:.2f}% | "
            f"clean={100*val['Clean_Acc']:.2f}% "
            f"fixedTargetPGDrob={100*val['Fixed_TargetPGD_Robust_Acc_CC']:.2f}% "
            f"fixedDiffrob={100*val['Fixed_TargetDiffusion_Robust_Acc_CC']:.2f}% "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_val = {**val, "Val_Loss": val_loss}
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            print(f"  -> BEST {spec.display_name} epoch={epoch}")
        clean_cuda()

    if best_state is None:
        raise RuntimeError(f"No best checkpoint for {spec.key}")

    for r in history:
        r["Selected_Checkpoint"] = bool(int(r["Epoch"]) == int(best_epoch))

    model.load_state_dict(best_state, strict=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.cpu().state_dict(),
        "normalizer": surrogate_norm.state_dict(),
        "training": "target_source_diffusion_diversity_offline_hardening_surrogate_amc",
        "experiment_key": spec.key,
        "offline_mode": spec.legacy_mode or spec.key,
        "display_name": spec.display_name,
        "experiment_role": spec.role,
        "variant_policy": spec.variant_policy,
        "stored_variants_per_clean": dataset.num_variants,
        "attack_bank_files": [str(p) for p in attack_bank_paths],
        "diversity_signature": diversity_signature,
        "defense_model_role": "surrogate",
        "defense_model_class": "SurrogateAMC",
        "attack_source_model_role": (
            "none_clean_control" if spec.key == "clean" else "original_target"
        ),
        "attacks_generated_against_moving_defense": False,
        "checkpoint_selection_uses_current_defense_gradients": False,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "best_validation": best_val,
        "full_epoch_training": True,
        "configured_epochs": int(args.epochs),
        "completed_epochs": len(history),
        "early_stopping_enabled": False,
        "cross_method_loss_metric": "Val_Clean_Loss",
        "train_epsilon_in_target_coordinate": args.train_eps,
        "adv_ratio": 0.0 if spec.key == "clean" else args.adv_ratio,
        "source_target_checkpoint": str(target_checkpoint_path),
        "source_target_sha256": target_hash,
        "initial_surrogate_checkpoint": str(surrogate_checkpoint_path),
        "initial_surrogate_sha256": surrogate_hash,
        "diffusion_checkpoint": (
            str(diffusion_path) if spec.key.startswith("diffusion") else None
        ),
        "diffusion_sha256": (
            diffusion_hash if spec.key.startswith("diffusion") else None
        ),
        "attack_coordinate_normalizer_role": (
            "target_amc" if spec.key != "clean" else None
        ),
        "defense_coordinate_normalizer_role": "surrogate_amc",
        "history": history,
    }
    common.save_artifact(ckpt, output_path)

    eval_model = common.SurrogateAMC().to(device)
    eval_model.load_state_dict(best_state, strict=True)
    freeze_model(eval_model)
    return eval_model, ckpt, pd.DataFrame(history)


@torch.no_grad()
def precompute_diffusion_unit_cache_sampler(
    shared,
    loader: DataLoader,
    attack_diffusion: nn.Module,
    attack_normalizer: common.GlobalRMSNormalizer,
    attack_betas: torch.Tensor,
    tanh_temperature: float,
    sampler: str,
    sampling_steps: int,
    ddim_eta: float,
    device: torch.device,
    seed: int,
):
    common.set_global_seed(int(seed))
    cache = []
    for physical, labels, _snrs in tqdm(
        loader,
        desc=f"Cache Target-Diffusion {sampler}:{sampling_steps}",
        leave=False,
    ):
        physical = physical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_attack = shared.normalize_torch(physical, attack_normalizer)
        if sampler == "ddpm_full":
            raw = shared.generate_delta_full_ddpm(
                model=attack_diffusion,
                clean=clean_attack,
                labels=labels,
                betas=attack_betas,
            )
        elif sampler == "ddim":
            raw = shared.generate_delta_fast_ddim(
                model=attack_diffusion,
                clean=clean_attack,
                labels=labels,
                betas=attack_betas,
                sampling_steps=int(sampling_steps),
                eta=float(ddim_eta),
            )
        else:
            raise ValueError(sampler)
        unit = torch.tanh(raw.float() / float(tanh_temperature)).unsqueeze(1)
        cache.append(unit.detach().cpu())
        del physical, labels, clean_attack, raw, unit
    clean_cuda()
    return cache


def evaluate_diverse_diffusion_transfer_suite(
    shared,
    evaluator,
    loader: DataLoader,
    systems: Sequence[Any],
    attack_diffusion: nn.Module,
    attack_normalizer: common.GlobalRMSNormalizer,
    attack_betas: torch.Tensor,
    tanh_temperature: float,
    ddim_steps: Sequence[int],
    ddim_eta: float,
    eps_values: Sequence[float],
    device: torch.device,
    seed: int,
    keep_ratio: float,
) -> pd.DataFrame:
    """Evaluate frozen diffusion transfer under several sampling depths."""
    configs = [("DDPM-Full", "ddpm_full", len(attack_betas))]
    configs.extend(
        (f"DDIM-{int(s)}", "ddim", int(s)) for s in ddim_steps
    )

    caches = {}
    for ci, (label, sampler, steps) in enumerate(configs):
        caches[label] = precompute_diffusion_unit_cache_sampler(
            shared=shared,
            loader=loader,
            attack_diffusion=attack_diffusion,
            attack_normalizer=attack_normalizer,
            attack_betas=attack_betas,
            tanh_temperature=tanh_temperature,
            sampler=sampler,
            sampling_steps=steps,
            ddim_eta=ddim_eta,
            device=device,
            seed=seed + 50_000 * (ci + 1),
        )

    rows = []
    for label, _sampler, _steps in configs:
        unit_cache = caches[label]
        for epsilon in eps_values:
            per_system = {
                system.name: {
                    "total": 0,
                    "clean_correct": 0,
                    "adv_correct": 0,
                    "success": 0,
                    "linf": 0.0,
                    "l2": 0.0,
                    "power": 0.0,
                    "psr": 0.0,
                    "ober": 0.0,
                    "perturb_n": 0,
                }
                for system in systems
            }

            for bi, (physical, labels, _snrs) in enumerate(loader):
                physical = physical.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                bank = evaluator.diffusion_bank_from_cached_unit(
                    shared=shared,
                    physical_clean=physical,
                    unit_bank_cpu=unit_cache[bi],
                    attack_normalizer=attack_normalizer,
                    epsilon=float(epsilon),
                )
                adv_phys = bank[:, 0]

                for system in systems:
                    clean_norm = shared.normalize_torch(
                        physical, system.normalizer
                    )
                    adv_norm = shared.normalize_torch(
                        adv_phys, system.normalizer
                    )
                    clean_pred = system.model(clean_norm).argmax(dim=1)
                    adv_pred = system.model(adv_norm).argmax(dim=1)
                    cc = clean_pred.eq(labels)
                    acc = per_system[system.name]
                    acc["total"] += len(labels)
                    acc["clean_correct"] += int(cc.sum().item())
                    acc["adv_correct"] += int(adv_pred.eq(labels).sum().item())
                    acc["success"] += int((cc & adv_pred.ne(labels)).sum().item())
                    metrics = evaluator.perturbation_stats_per_sample(
                        clean_norm,
                        adv_norm,
                        keep_ratio=keep_ratio,
                    )
                    linf, l2, power, psr, ober = metrics
                    acc["perturb_n"] += len(labels)
                    acc["linf"] += float(linf.sum().item())
                    acc["l2"] += float(l2.sum().item())
                    acc["power"] += float(power.sum().item())
                    acc["psr"] += float(psr.sum().item())
                    acc["ober"] += float(ober.sum().item())
                del physical, labels, bank, adv_phys

            for system in systems:
                acc = per_system[system.name]
                total = max(acc["total"], 1)
                cc = max(acc["clean_correct"], 1)
                pn = max(acc["perturb_n"], 1)
                asr = acc["success"] / cc
                rows.append({
                    "Defense": system.name,
                    "Defense_Family": system.family,
                    "Attack": f"PureDiffusion-{label}",
                    "Threat_Model": "Frozen-Generator Transfer",
                    "Attack_Source": "Target-trained Diffusion",
                    "EPS": float(epsilon),
                    "Samples": int(acc["total"]),
                    "Clean_Acc": acc["clean_correct"] / total,
                    "Adv_Acc": acc["adv_correct"] / total,
                    "ASR_on_CleanCorrect": asr,
                    "Robust_Acc_on_CleanCorrect": 1.0 - asr,
                    "Mean_Linf": acc["linf"] / pn,
                    "Mean_L2": acc["l2"] / pn,
                    "Mean_Power": acc["power"] / pn,
                    "Mean_PSR_dB": acc["psr"] / pn,
                    "Mean_OBER": acc["ober"] / pn,
                })

    return pd.DataFrame(rows)


def build_all_vs_pgd_comparison(
    global_df: pd.DataFrame,
    pgd_display: str,
) -> pd.DataFrame:
    """Compare every completed defense with PGD-Offline.

    White-box attacks are crafted separately against each CURRENT defense, so
    `Attack_Source` is intentionally different for every defense and MUST NOT be
    used as a merge key.  Transfer/frozen-generator attacks, on the other hand,
    share a real external source and therefore keep Attack_Source in the key.
    """
    attack_df = global_df[global_df["Attack"] != "Clean"].copy()
    baseline = attack_df[attack_df["Defense"] == pgd_display].copy()
    rows: List[Dict[str, Any]] = []

    for defense in sorted(attack_df["Defense"].dropna().unique()):
        if defense == pgd_display:
            continue
        proposed = attack_df[attack_df["Defense"] == defense].copy()

        # 1) Current-defense white-box: pair by attack family and epsilon only.
        prop_wb = proposed[proposed["Threat_Model"].eq("White-Box")].copy()
        base_wb = baseline[baseline["Threat_Model"].eq("White-Box")].copy()
        wb_join = ["Attack", "Threat_Model", "EPS"]
        merged_wb = prop_wb.merge(
            base_wb,
            on=wb_join,
            suffixes=("_Method", "_PGD"),
            how="inner",
            validate="one_to_one",
        )
        for _, r in merged_wb.iterrows():
            method_asr = float(r["ASR_on_CleanCorrect_Method"])
            pgd_asr = float(r["ASR_on_CleanCorrect_PGD"])
            rows.append({
                "Method": defense,
                "Attack": r["Attack"],
                "Threat_Model": r["Threat_Model"],
                "Attack_Source": "CurrentDefense (paired white-box)",
                "Method_Attack_Source": r.get("Attack_Source_Method", defense),
                "PGDOffline_Attack_Source": r.get("Attack_Source_PGD", pgd_display),
                "EPS": float(r["EPS"]),
                "Method_ASR": method_asr,
                "PGDOffline_ASR": pgd_asr,
                "ASR_Improvement_pp_MethodMinusPGD": 100.0 * (pgd_asr - method_asr),
                "Method_RobustAcc_CC": float(r["Robust_Acc_on_CleanCorrect_Method"]),
                "PGDOffline_RobustAcc_CC": float(r["Robust_Acc_on_CleanCorrect_PGD"]),
            })

        # 2) Transfer / frozen-generator attacks: source identity must match.
        prop_other = proposed[~proposed["Threat_Model"].eq("White-Box")].copy()
        base_other = baseline[~baseline["Threat_Model"].eq("White-Box")].copy()
        other_join = ["Attack", "Threat_Model", "Attack_Source", "EPS"]
        merged_other = prop_other.merge(
            base_other,
            on=other_join,
            suffixes=("_Method", "_PGD"),
            how="inner",
        )
        for _, r in merged_other.iterrows():
            method_asr = float(r["ASR_on_CleanCorrect_Method"])
            pgd_asr = float(r["ASR_on_CleanCorrect_PGD"])
            rows.append({
                "Method": defense,
                "Attack": r["Attack"],
                "Threat_Model": r["Threat_Model"],
                "Attack_Source": r["Attack_Source"],
                "Method_Attack_Source": r["Attack_Source"],
                "PGDOffline_Attack_Source": r["Attack_Source"],
                "EPS": float(r["EPS"]),
                "Method_ASR": method_asr,
                "PGDOffline_ASR": pgd_asr,
                "ASR_Improvement_pp_MethodMinusPGD": 100.0 * (pgd_asr - method_asr),
                "Method_RobustAcc_CC": float(r["Robust_Acc_on_CleanCorrect_Method"]),
                "PGDOffline_RobustAcc_CC": float(r["Robust_Acc_on_CleanCorrect_PGD"]),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "Method", "Attack", "Threat_Model", "Attack_Source",
            "Method_Attack_Source", "PGDOffline_Attack_Source", "EPS",
            "Method_ASR", "PGDOffline_ASR",
            "ASR_Improvement_pp_MethodMinusPGD",
            "Method_RobustAcc_CC", "PGDOffline_RobustAcc_CC",
        ])

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["Threat_Model", "Attack", "EPS", "Method"],
        kind="stable",
    ).reset_index(drop=True)


def build_focus_summary(
    global_df: pd.DataFrame,
    focus_eps: float,
) -> pd.DataFrame:
    clean = (
        global_df[global_df["Attack"] == "Clean"]
        .set_index("Defense")["Clean_Acc"]
        .to_dict()
    )
    wb = global_df[
        global_df["EPS"].notna()
        & np.isclose(global_df["EPS"].fillna(-999).to_numpy(float), focus_eps)
        & global_df["Threat_Model"].eq("White-Box")
        & global_df["Attack"].isin(["FGSM", "BIM10", "PGD10", "PGD20"])
    ]
    rows = []
    for defense, group in wb.groupby("Defense"):
        rows.append({
            "Defense": defense,
            "Clean_Acc": clean.get(defense, float("nan")),
            "Mean_WhiteBox_ASR": float(group["ASR_on_CleanCorrect"].mean()),
            "Mean_WhiteBox_RobustAcc_CC": float(
                group["Robust_Acc_on_CleanCorrect"].mean()
            ),
            "PGD10_ASR": float(
                group.loc[group["Attack"].eq("PGD10"), "ASR_on_CleanCorrect"].iloc[0]
            ) if bool(group["Attack"].eq("PGD10").any()) else float("nan"),
            "PGD20_ASR": float(
                group.loc[group["Attack"].eq("PGD20"), "ASR_on_CleanCorrect"].iloc[0]
            ) if bool(group["Attack"].eq("PGD20").any()) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(
        ["Mean_WhiteBox_ASR", "Clean_Acc"],
        ascending=[True, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable Target-source offline hardening of SurrogateAMC with "
            "diffusion diversity: multi-start, multi-DDIM-depth and combined banks."
        )
    )
    parser.add_argument("--training-script", type=Path, default=None)
    parser.add_argument("--evaluator-script", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2040)

    # Source attack generation / legacy baseline compatibility.
    parser.add_argument("--offline-data-ratio", type=float, default=1.0)
    parser.add_argument("--train-eps", type=float, default=0.10)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument(
        "--diffusion-k",
        type=int,
        default=1,
        help="Legacy K1 baseline setting. Keep at 1; diversity uses --multistart-k.",
    )
    parser.add_argument("--generation-batch-size", type=int, default=128)

    # New diversity controls.
    parser.add_argument("--multistart-k", type=int, default=4)
    parser.add_argument(
        "--diverse-ddim-steps",
        nargs="+",
        type=int,
        default=[2, 4, 6, 8],
        help=(
            "Reverse sampling depths used to create a multi-depth DDIM attack bank. "
            "Each value is a number of denoising network calls, not early stopping."
        ),
    )
    parser.add_argument("--diverse-ddim-eta", type=float, default=0.0)
    parser.add_argument(
        "--diversity-stat-max-samples",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--diffusion-checkpoint",
        type=Path,
        default=(
            common.ARTIFACT_ROOT
            / "models"
            / "adversarial_diffusion_unsupervised_0909.pt"
        ),
    )
    parser.add_argument(
        "--allow-uncertain-diffusion-provenance",
        action="store_true",
    )
    parser.add_argument(
        "--legacy-diffusion-tanh-temperature",
        type=float,
        default=None,
    )

    # Training / resume.
    default_experiments = [
        "clean",
        "pgd",
        "diffusion_k1",
        "diffusion_multistart_cycle",
        "diffusion_multistep_cycle",
        "diffusion_diverse_cycle",
    ]
    all_choices = [
        "clean",
        "pgd",
        "diffusion_k1",
        "diffusion_multistart_cycle",
        "diffusion_multistep_cycle",
        "diffusion_diverse_cycle",
    ]
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=all_choices,
        default=default_experiments,
        help=(
            "Experiments included in this run. By default --retrain-all-models retrains "
            "all selected models while reusing compatible stored attack banks."
        ),
    )
    parser.add_argument(
        "--force-experiments",
        nargs="*",
        choices=all_choices,
        default=[],
        help="Force selected model experiments to retrain even if compatible checkpoints exist.",
    )
    parser.add_argument(
        "--force-regenerate-banks",
        nargs="*",
        choices=["baseline_diffusion", "pgd", "multistart", "multistep"],
        default=[],
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse compatible attack banks and, when model retraining is disabled, checkpoints/evaluation.",
    )
    parser.add_argument(
        "--retrain-all-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Default: retrain every selected defense model from the same original "
            "SurrogateAMC initialization. Existing attack banks are still reused."
        ),
    )
    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adv-ratio", type=float, default=0.5)

    # Preserve the same fixed validation rule as the previous experiment so the
    # retrained methods remain directly comparable; losses are now recorded too.
    parser.add_argument("--validation-max-samples", type=int, default=2048)
    parser.add_argument("--validation-pgd-steps", type=int, default=10)
    parser.add_argument("--checkpoint-clean-weight", type=float, default=0.20)
    parser.add_argument("--checkpoint-target-pgd-weight", type=float, default=0.40)
    parser.add_argument("--checkpoint-diffusion-weight", type=float, default=0.40)

    # Unified final evaluation.
    parser.add_argument(
        "--eval-eps-values",
        nargs="+",
        type=float,
        default=[0.03, 0.05, 0.10, 0.20, 0.30],
    )
    parser.add_argument(
        "--classical-attacks",
        nargs="+",
        choices=["FGSM", "BIM10", "PGD2", "PGD10", "PGD20"],
        default=["FGSM", "BIM10", "PGD10", "PGD20"],
    )
    parser.add_argument("--test-sample-fraction", type=float, default=0.10)
    parser.add_argument("--test-max-samples", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--keep-ratio", type=float, default=0.25)
    parser.add_argument(
        "--diverse-diffusion-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also test frozen Target-Diffusion under DDPM-full and every DDIM depth.",
    )
    parser.add_argument("--force-unified-eval", action="store_true")
    parser.add_argument("--skip-unified-eval", action="store_true")

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            common.ARTIFACT_ROOT
            / "offline_targetsource_to_surrogate_hardening_0909"
        ),
        help=(
            "Defaults to the previous experiment root so expensive PGD/Diffusion "
            "attack banks are reused. With --retrain-all-models (default), defense "
            "checkpoints are retrained and overwritten, then evaluated afresh."
        ),
    )
    args = parser.parse_args()

    if args.eval_only:
        # Evaluation-only must load existing checkpoints rather than retrain them.
        args.retrain_all_models = False
    if args.retrain_all_models and not args.skip_unified_eval:
        # The user explicitly requested a fresh unified comparison after retraining.
        args.force_unified_eval = True

    # ----------------------------- validation -----------------------------
    if not 0 < args.offline_data_ratio <= 1:
        parser.error("--offline-data-ratio must be in (0,1]")
    if args.train_eps <= 0:
        parser.error("--train-eps must be positive")
    if args.diffusion_k != 1:
        parser.error("Keep --diffusion-k=1 for the legacy baseline; use --multistart-k for diversity.")
    if args.multistart_k < 2:
        parser.error("--multistart-k must be >= 2")
    if not args.diverse_ddim_steps:
        parser.error("--diverse-ddim-steps cannot be empty")
    if len(set(args.diverse_ddim_steps)) != len(args.diverse_ddim_steps):
        parser.error("--diverse-ddim-steps must not contain duplicates")
    if any(x < 1 for x in args.diverse_ddim_steps):
        parser.error("--diverse-ddim-steps must be >= 1")
    if any(x <= 0 for x in args.eval_eps_values):
        parser.error("--eval-eps-values must be positive")
    if args.pgd_steps < 1 or args.validation_pgd_steps < 1:
        parser.error("PGD steps must be >= 1")
    if min(args.generation_batch_size, args.train_batch_size, args.eval_batch_size) < 1:
        parser.error("batch sizes must be positive")
    if not 0 <= args.adv_ratio <= 1:
        parser.error("--adv-ratio must be in [0,1]")
    if not 0 < args.test_sample_fraction <= 1:
        parser.error("--test-sample-fraction must be in (0,1]")

    # -------------------------- dynamic utilities --------------------------
    training_script = resolve_training_script(args.training_script)
    evaluator_script = resolve_evaluator_script(args.evaluator_script)
    shared = _import_module(training_script, "ad3_diversity_shared")
    evaluator = _import_module(evaluator_script, "ad7r_diversity_eval")

    required_shared = [
        "gradient_attack_physical",
        "generate_delta_full_ddpm",
        "generate_delta_fast_ddim",
        "make_beta_schedule",
        "normalize_torch",
        "denormalize_torch",
        "PhysicalIQDataset",
        "stratified_fraction_sample",
        "balanced_limit_by_snr",
    ]
    missing = [x for x in required_shared if not hasattr(shared, x)]
    if missing:
        raise RuntimeError("Selected AD-3 utility is missing: " + ", ".join(missing))

    required_eval = [
        "DefenseSystem",
        "perturbation_stats_per_sample",
        "evaluate_clean",
        "evaluate_epsilon_sweep",
        "accumulators_to_frames",
        "diffusion_bank_from_cached_unit",
    ]
    missing = [x for x in required_eval if not hasattr(evaluator, x)]
    if missing:
        raise RuntimeError("Selected AD-7R evaluator is missing: " + ", ".join(missing))

    common.set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset_dir = args.output_root / "datasets"
    model_dir = args.output_root / "models"
    log_dir = args.output_root / "logs"
    for p in [dataset_dir, model_dir, log_dir]:
        p.mkdir(parents=True, exist_ok=True)

    encoder = common.make_label_encoder()

    # ---------------------- fixed source / defense init --------------------
    target_checkpoint_path = Path(common.model_path("target"))
    surrogate_checkpoint_path = Path(common.model_path("surrogate"))
    if not target_checkpoint_path.exists():
        raise FileNotFoundError(target_checkpoint_path)
    if not surrogate_checkpoint_path.exists():
        raise FileNotFoundError(surrogate_checkpoint_path)

    target_ckpt = common.load_artifact(target_checkpoint_path)
    target_norm = load_role_normalizer(target_ckpt, "target", "TargetAMC")
    source_target = freeze_model(clone_target(target_ckpt, device))

    surrogate_ckpt = common.load_artifact(surrogate_checkpoint_path)
    surrogate_norm = load_role_normalizer(surrogate_ckpt, "surrogate", "SurrogateAMC")
    original_surrogate = freeze_model(clone_surrogate(surrogate_ckpt, device))

    diffusion_path = args.diffusion_checkpoint.expanduser().resolve()
    if not diffusion_path.exists():
        raise FileNotFoundError(diffusion_path)
    diff_ckpt = common.load_artifact(diffusion_path)

    explicit_guide = diff_ckpt.get("guide_model_role")
    normalizer_role = str(diff_ckpt.get("normalizer_role", "")).lower()
    training_tag = str(diff_ckpt.get("training", "")).lower()
    if explicit_guide is not None and str(explicit_guide).lower() not in {
        "target", "target_amc", "original_target"
    }:
        raise RuntimeError(
            f"Diffusion checkpoint records non-Target guide={explicit_guide!r}."
        )
    provenance_target = (
        explicit_guide is not None
        or normalizer_role in {"target", "target_amc"}
        or "target_guided" in training_tag
    )
    if not provenance_target and not args.allow_uncertain_diffusion_provenance:
        raise RuntimeError(
            "Diffusion checkpoint metadata does not prove TargetAMC provenance. "
            "Use --allow-uncertain-diffusion-provenance only after manual verification."
        )

    diffusion = common.AdversarialConditionalNoisePredictor(
        num_classes=len(encoder.classes_)
    ).to(device)
    diffusion.load_state_dict(diff_ckpt["model_state"], strict=True)
    freeze_model(diffusion)
    if "normalizer" not in diff_ckpt:
        raise RuntimeError("Diffusion checkpoint has no normalizer.")
    diffusion_norm = common.GlobalRMSNormalizer.from_state_dict(diff_ckpt["normalizer"])
    target_diff_norm_gap = abs(
        float(diffusion_norm.scale) - float(target_norm.scale)
    ) / max(abs(float(target_norm.scale)), 1e-30)
    if target_diff_norm_gap > 1e-6:
        raise RuntimeError(
            f"Diffusion/Target normalizer mismatch={target_diff_norm_gap:.3%}."
        )

    if "tanh_temperature" in diff_ckpt:
        diffusion_temperature = float(diff_ckpt["tanh_temperature"])
    elif args.legacy_diffusion_tanh_temperature is not None:
        diffusion_temperature = float(args.legacy_diffusion_tanh_temperature)
    else:
        raise RuntimeError(
            "Diffusion checkpoint lacks tanh_temperature. Pass "
            "--legacy-diffusion-tanh-temperature only for a verified legacy checkpoint."
        )
    if diffusion_temperature <= 0:
        raise RuntimeError("Diffusion tanh_temperature must be positive")

    diffusion_steps = int(diff_ckpt.get("diffusion_steps", 10))
    if any(int(x) > diffusion_steps for x in args.diverse_ddim_steps):
        parser.error(
            f"--diverse-ddim-steps cannot exceed diffusion_steps={diffusion_steps}"
        )
    diffusion_betas = shared.make_beta_schedule(diffusion_steps, device)

    target_hash = sha256_file(target_checkpoint_path)
    surrogate_hash = sha256_file(surrogate_checkpoint_path)
    diffusion_hash = sha256_file(diffusion_path)

    specs = build_experiment_specs(
        seed=args.seed,
        multistart_k=args.multistart_k,
        ddim_steps=args.diverse_ddim_steps,
    )
    diversity_signature = json.dumps({
        "multistart_k": int(args.multistart_k),
        "ddim_steps": [int(x) for x in args.diverse_ddim_steps],
        "ddim_eta": float(args.diverse_ddim_eta),
        "diffusion_hash": diffusion_hash,
        "tanh_temperature": diffusion_temperature,
        "train_eps": float(args.train_eps),
    }, sort_keys=True)

    print("=" * 124)
    print(" TARGET-SOURCE OFFLINE HARDENING — DIFFUSION DIVERSITY STUDY")
    print("=" * 124)
    print(f"[INFO] device                     : {device}")
    print("[INFO] attack source              : Original TargetAMC")
    print("[INFO] defense architecture       : SurrogateAMC")
    print(f"[INFO] train eps                  : {args.train_eps}")
    print(f"[INFO] existing Diffusion         : {diffusion_path}")
    print(f"[INFO] multi-start K              : {args.multistart_k}")
    print(f"[INFO] DDIM depth variants        : {list(args.diverse_ddim_steps)}")
    print(f"[INFO] required experiments       : {list(args.experiments)}")
    print(f"[INFO] auto resume / bank reuse    : {args.auto_resume}")
    print(f"[INFO] retrain all selected models : {args.retrain_all_models}")
    print("[INFO] all adversarial experiments use cycle policy: exactly N pairs/epoch.")
    print("=" * 124)

    # -------------------------- common source rows -------------------------
    diff_corpus = common.load_artifact(common.corpus_path("diff"))
    train_df = diff_corpus["splits"]["train"].reset_index(drop=True)
    val_df = diff_corpus["splits"]["val"].reset_index(drop=True)
    del diff_corpus

    source_df = stratified_source_sample(
        train_df, args.offline_data_ratio, args.seed + 11
    )
    _, robust_val_df = split_calibration_and_validation(val_df, args.seed + 101)
    if args.validation_max_samples > 0 and len(robust_val_df) > args.validation_max_samples:
        robust_val_df = shared.balanced_limit_by_snr(
            robust_val_df, args.validation_max_samples, args.seed + 202
        )
    del train_df, val_df
    gc.collect()

    source_dataset = IndexedPhysicalDataset(source_df, encoder)
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.generation_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    source_shape = tuple(source_dataset.iq.shape)
    tail_shape = tuple(source_shape[1:])

    # Manifest remains compatible with the previous experiment root.
    manifest_cols = [
        c for c in ["_Original_Row_Index", "Sample_ID", "Modulation_Label", "SNR_dB"]
        if c in source_df.columns
    ]
    manifest = source_df[manifest_cols].copy()
    manifest["Offline_Train_EPS_TargetCoordinate"] = args.train_eps
    manifest["Attack_Source_AMC"] = "Original TargetAMC"
    manifest["Defense_AMC"] = "SurrogateAMC"
    manifest.to_csv(dataset_dir / "paired_source_manifest.csv", index=False)

    legacy_paths = build_cache_paths(
        root=dataset_dir,
        eps=args.train_eps,
        pgd_steps=args.pgd_steps,
        diffusion_k=1,
        n=len(source_dataset),
        seed=args.seed,
        target_hash=target_hash,
        diffusion_hash=diffusion_hash,
    )
    div_paths = diversity_cache_paths(
        root=dataset_dir,
        eps=args.train_eps,
        multistart_k=args.multistart_k,
        ddim_steps=args.diverse_ddim_steps,
        n=len(source_dataset),
        seed=args.seed,
        target_hash=target_hash,
        diffusion_hash=diffusion_hash,
    )
    bank_paths = {
        "baseline_diffusion": legacy_paths["diffusion"],
        "pgd": legacy_paths["pgd"],
        **div_paths,
    }
    expected_variants = {
        "baseline_diffusion": 1,
        "pgd": 1,
        "multistart": int(args.multistart_k),
        "multistep": len(args.diverse_ddim_steps),
    }

    # Pre-scan model checkpoints BEFORE generating attacks/validation caches.
    # This is the core resume behavior: a completed compatible experiment does
    # not require its old attack bank or validation attacks to be regenerated.
    output_paths = {
        key: model_dir / spec.checkpoint_filename
        for key, spec in specs.items()
    }
    precompleted_ckpts: Dict[str, dict] = {}
    if args.retrain_all_models:
        print("[RETRAIN] All selected defense checkpoints will be retrained from original SurrogateAMC.")
    elif args.auto_resume:
        for key in args.experiments:
            if key in args.force_experiments:
                continue
            path = output_paths[key]
            if not path.exists():
                continue
            try:
                ckpt = common.load_artifact(path)
                compatible, reason = checkpoint_is_compatible(
                    ckpt,
                    specs[key],
                    args,
                    target_hash,
                    surrogate_hash,
                    diffusion_hash,
                    diversity_signature,
                )
                if compatible:
                    precompleted_ckpts[key] = ckpt
                    print(
                        f"[RESUME] completed experiment detected -> "
                        f"{specs[key].display_name}: {path}"
                    )
                else:
                    print(
                        f"[WARN] Existing model is not compatible ({reason}) -> {path}"
                    )
            except Exception as exc:
                print(f"[WARN] Could not validate checkpoint {path}: {exc}")

    missing_requested = [
        key for key in args.experiments
        if key not in precompleted_ckpts
    ]
    if args.eval_only and missing_requested:
        missing_text = ", ".join(missing_requested)
        raise FileNotFoundError(
            f"--eval-only requested but compatible models are missing: {missing_text}"
        )

    print(f"[INFO] completed requested experiments : {sorted(precompleted_ckpts)}")
    print(f"[INFO] experiments still to run        : {missing_requested}")

    # Only missing experiments are allowed to request/generate attack banks.
    required_bank_keys = set()
    for key in missing_requested:
        required_bank_keys.update(specs[key].attack_bank_keys)

    if not args.eval_only:
        for bank_key in ["baseline_diffusion", "pgd", "multistart", "multistep"]:
            if bank_key not in required_bank_keys:
                continue
            path = bank_paths[bank_key]
            valid = valid_attack_bank(
                path,
                n=len(source_dataset),
                tail_shape=tail_shape,
                variants=expected_variants[bank_key],
            )
            force = bank_key in args.force_regenerate_banks
            if valid and not force:
                print(f"[RESUME] attack bank exists -> {bank_key}: {path}")
                continue
            if args.auto_resume and path.exists() and not valid:
                print(f"[WARN] Existing bank is incompatible and will be regenerated: {path}")

            if bank_key == "baseline_diffusion":
                generate_fixed_corpus(
                    mode="diffusion",
                    out_path=path,
                    loader=source_loader,
                    source_shape=source_shape,
                    source_target=source_target,
                    target_norm=target_norm,
                    shared=shared,
                    diffusion=diffusion,
                    diffusion_norm=diffusion_norm,
                    diffusion_betas=diffusion_betas,
                    diffusion_temperature=diffusion_temperature,
                    epsilon=args.train_eps,
                    pgd_steps=args.pgd_steps,
                    diffusion_k=1,
                    diffusion_sampler="ddpm_full",
                    diffusion_sampling_steps=diffusion_steps,
                    diffusion_ddim_eta=0.0,
                    device=device,
                    seed=args.seed + 1_000_000,
                )
            elif bank_key == "pgd":
                generate_fixed_corpus(
                    mode="pgd",
                    out_path=path,
                    loader=source_loader,
                    source_shape=source_shape,
                    source_target=source_target,
                    target_norm=target_norm,
                    shared=shared,
                    diffusion=diffusion,
                    diffusion_norm=diffusion_norm,
                    diffusion_betas=diffusion_betas,
                    diffusion_temperature=diffusion_temperature,
                    epsilon=args.train_eps,
                    pgd_steps=args.pgd_steps,
                    diffusion_k=1,
                    diffusion_sampler="ddpm_full",
                    diffusion_sampling_steps=diffusion_steps,
                    diffusion_ddim_eta=0.0,
                    device=device,
                    seed=args.seed + 2_000_000,
                )
            elif bank_key == "multistart":
                generate_multistart_diffusion_bank(
                    out_path=path,
                    loader=source_loader,
                    source_shape=source_shape,
                    shared=shared,
                    diffusion=diffusion,
                    diffusion_norm=diffusion_norm,
                    diffusion_betas=diffusion_betas,
                    diffusion_temperature=diffusion_temperature,
                    epsilon=args.train_eps,
                    multistart_k=args.multistart_k,
                    device=device,
                    seed=args.seed + 3_000_000,
                )
            elif bank_key == "multistep":
                generate_multistep_diffusion_bank(
                    out_path=path,
                    loader=source_loader,
                    source_shape=source_shape,
                    shared=shared,
                    diffusion=diffusion,
                    diffusion_norm=diffusion_norm,
                    diffusion_betas=diffusion_betas,
                    diffusion_temperature=diffusion_temperature,
                    epsilon=args.train_eps,
                    ddim_steps=args.diverse_ddim_steps,
                    ddim_eta=args.diverse_ddim_eta,
                    device=device,
                    seed=args.seed + 3_500_000,
                )
    else:
        for bank_key in required_bank_keys:
            if not valid_attack_bank(
                bank_paths[bank_key],
                len(source_dataset),
                tail_shape,
                expected_variants[bank_key],
            ):
                raise FileNotFoundError(
                    f"--eval-only requested but required bank is missing/incompatible: {bank_paths[bank_key]}"
                )

    # ------------------------ diversity diagnostics ------------------------
    diversity_rows = []
    for bank_key in ["baseline_diffusion", "multistart", "multistep"]:
        path = bank_paths[bank_key]
        if valid_attack_bank(
            path,
            len(source_dataset),
            tail_shape,
            expected_variants[bank_key],
        ):
            diversity_rows.append(
                compute_attack_bank_diversity(
                    bank_name=bank_key,
                    path=path,
                    clean_iq=source_dataset.iq,
                    normalizer=target_norm,
                    max_samples=args.diversity_stat_max_samples,
                    seed=args.seed + 909,
                )
            )
    diversity_df = pd.DataFrame(diversity_rows)
    diversity_df.to_csv(log_dir / "attack_bank_diversity.csv", index=False)
    if not diversity_df.empty:
        print("\n[INFO] Attack-bank diversity diagnostics:")
        print(diversity_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Aggregate attack strength / transfer for stored banks. This is diagnostic;
    # it does not affect checkpoint selection. Existing rows are reused so a
    # rerun does not spend time rescoring already-completed attack corpora.
    generation_summary_path = dataset_dir / "generation_summary_diversity.csv"
    old_diversity_summary = (
        pd.read_csv(generation_summary_path)
        if generation_summary_path.exists()
        else pd.DataFrame()
    )
    old_legacy_summary_path = dataset_dir / "generation_summary.csv"
    old_legacy_summary = (
        pd.read_csv(old_legacy_summary_path)
        if old_legacy_summary_path.exists()
        else pd.DataFrame()
    )

    generation_rows = []
    for bank_key in ["baseline_diffusion", "pgd", "multistart", "multistep"]:
        path = bank_paths[bank_key]
        if not valid_attack_bank(
            path,
            len(source_dataset),
            tail_shape,
            expected_variants[bank_key],
        ):
            continue
        attack_name = {
            "baseline_diffusion": "TargetDiffusionK1",
            "pgd": f"TargetPGD{args.pgd_steps}",
            "multistart": f"TargetDiffusionMultiStartK{args.multistart_k}",
            "multistep": "TargetDiffusionDDIMDepthBank",
        }[bank_key]

        for eval_name, eval_model, eval_norm in [
            ("Source-OriginalTarget", source_target, target_norm),
            ("Transfer-OriginalSurrogate", original_surrogate, surrogate_norm),
        ]:
            reused = None
            if not old_diversity_summary.empty and {
                "Bank_Key", "Evaluation_Model", "Bank_Path"
            }.issubset(old_diversity_summary.columns):
                hit = old_diversity_summary[
                    old_diversity_summary["Bank_Key"].eq(bank_key)
                    & old_diversity_summary["Evaluation_Model"].eq(eval_name)
                    & old_diversity_summary["Bank_Path"].eq(str(path))
                ]
                if len(hit) == 1:
                    reused = hit.iloc[0].to_dict()

            # Seed the new diversity summary from the previous AD-5T baseline
            # summary when available, avoiding a full 103,680-sample rescore.
            if (
                reused is None
                and bank_key in {"baseline_diffusion", "pgd"}
                and not old_legacy_summary.empty
                and {"Attack", "Evaluation_Model"}.issubset(old_legacy_summary.columns)
            ):
                hit = old_legacy_summary[
                    old_legacy_summary["Attack"].eq(attack_name)
                    & old_legacy_summary["Evaluation_Model"].eq(eval_name)
                ]
                if len(hit) == 1:
                    reused = hit.iloc[0].to_dict()
                    reused["Stored_Variants_Per_Clean"] = 1

            if reused is not None:
                reused["Bank_Key"] = bank_key
                reused["Bank_Path"] = str(path)
                generation_rows.append(reused)
                print(f"[RESUME] generation score reused -> {bank_key} / {eval_name}")
                continue

            row = score_cached_attack_bank(
                attack_name=attack_name,
                path=path,
                loader=source_loader,
                evaluation_model=eval_model,
                evaluation_norm=eval_norm,
                evaluation_model_name=eval_name,
                evaluator=evaluator,
                epsilon=args.train_eps,
                device=device,
            )
            row["Bank_Key"] = bank_key
            row["Bank_Path"] = str(path)
            generation_rows.append(row)

    pd.DataFrame(generation_rows).to_csv(
        generation_summary_path,
        index=False,
    )

    # ------------------------ fixed validation cache -----------------------
    # Build these only when at least one model really needs training.
    robust_val_dataset = shared.PhysicalIQDataset(robust_val_df, encoder)
    robust_val_loader = DataLoader(
        robust_val_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    if missing_requested:
        fixed_val_diff = cache_validation_diffusion(
            shared=shared,
            loader=robust_val_loader,
            diffusion=diffusion,
            diffusion_norm=diffusion_norm,
            betas=diffusion_betas,
            tanh_temperature=diffusion_temperature,
            epsilon=args.train_eps,
            device=device,
            seed=args.seed + 4_000_000,
        )
        fixed_val_pgd = cache_validation_target_pgd(
            shared=shared,
            loader=robust_val_loader,
            source_target=source_target,
            target_norm=target_norm,
            epsilon=args.train_eps,
            pgd_steps=args.validation_pgd_steps,
            device=device,
            seed=args.seed + 4_100_000,
        )
    else:
        fixed_val_diff = None
        fixed_val_pgd = None
        print("[RESUME] No missing training experiments; fixed validation attacks are not regenerated.")

    # ---------------------- train only missing models ----------------------
    completed_ckpts: Dict[str, dict] = dict(precompleted_ckpts)
    history_frames = []

    # Source TargetAMC / attacker / original Surrogate are no longer needed while
    # optimizing new defenses. Move them off GPU so diversity experiments retain
    # the same comfortable memory headroom as the previous single-model run.
    if missing_requested:
        source_target = freeze_model(source_target.cpu())
        diffusion = freeze_model(diffusion.cpu())
        original_surrogate = freeze_model(original_surrogate.cpu())
        clean_cuda()

    for key in args.experiments:
        if key in completed_ckpts:
            continue
        spec = specs[key]
        output_path = output_paths[key]
        attack_paths = [bank_paths[x] for x in spec.attack_bank_keys]
        for p in attack_paths:
            if not p.exists():
                raise FileNotFoundError(p)

        common.set_global_seed(args.seed)
        model, ckpt, history = train_surrogate_experiment(
            spec=spec,
            surrogate_ckpt=surrogate_ckpt,
            surrogate_norm=surrogate_norm,
            clean_iq=source_dataset.iq,
            labels=source_dataset.labels,
            attack_bank_paths=attack_paths,
            val_loader=robust_val_loader,
            fixed_val_diff=fixed_val_diff,
            fixed_val_pgd=fixed_val_pgd,
            shared=shared,
            device=device,
            output_path=output_path,
            args=args,
            target_checkpoint_path=target_checkpoint_path,
            target_hash=target_hash,
            surrogate_checkpoint_path=surrogate_checkpoint_path,
            surrogate_hash=surrogate_hash,
            diffusion_path=diffusion_path,
            diffusion_hash=diffusion_hash,
            diversity_signature=diversity_signature,
        )
        completed_ckpts[key] = ckpt
        history_frames.append(history)
        del model
        clean_cuda()

    # Also discover compatible completed known experiments even if they were not
    # explicitly requested this invocation, so unified evaluation uses every
    # completed model under this experiment root.
    for key in args.experiments:
        spec = specs[key]
        if key in completed_ckpts:
            continue
        path = output_paths[key]
        if not path.exists():
            continue
        try:
            ckpt = common.load_artifact(path)
            compatible, _reason = checkpoint_is_compatible(
                ckpt, spec, args, target_hash, surrogate_hash, diffusion_hash,
                diversity_signature,
            )
            if compatible:
                completed_ckpts[key] = ckpt
                print(f"[DISCOVER] completed model added to unified eval: {spec.display_name}")
        except Exception:
            pass

    if history_frames:
        new_history = pd.concat(history_frames, ignore_index=True)
        history_path = log_dir / "training_history_diversity.csv"
        retrained_keys = set(new_history["Experiment_Key"].astype(str).unique())
        if history_path.exists():
            old_history = pd.read_csv(history_path)
            if "Experiment_Key" in old_history.columns:
                old_history = old_history[
                    ~old_history["Experiment_Key"].astype(str).isin(retrained_keys)
                ]
            new_history = pd.concat([old_history, new_history], ignore_index=True)
        subset = [c for c in ["Experiment_Key", "Epoch"] if c in new_history.columns]
        if subset:
            new_history = new_history.drop_duplicates(subset=subset, keep="last")
        new_history = new_history.sort_values(
            ["Experiment_Key", "Epoch"], kind="stable"
        ).reset_index(drop=True)
        new_history.to_csv(history_path, index=False)

        loss_cols = [
            "Experiment_Key", "Mode", "Epoch", "Train_Loss",
            "Val_Clean_Loss", "Val_TargetPGD_Loss",
            "Val_TargetDiffusion_Loss", "Val_Loss",
            "Train_Acc", "Clean_Acc",
            "Fixed_TargetPGD_Robust_Acc_CC",
            "Fixed_TargetDiffusion_Robust_Acc_CC",
            "Checkpoint_Score", "Selected_Checkpoint",
        ]
        present_loss_cols = [c for c in loss_cols if c in new_history.columns]
        new_history[present_loss_cols].to_csv(
            log_dir / "loss_history.csv", index=False
        )

    # Access protocol records all completed methods.
    protocol_rows = []
    for key, ckpt in completed_ckpts.items():
        spec = specs[key]
        protocol_rows.append({
            "Experiment_Key": key,
            "Defense_Method": spec.display_name,
            "Role": spec.role,
            "Variant_Policy": spec.variant_policy,
            "Stored_Variants_Per_Clean": ckpt.get("stored_variants_per_clean", 0 if key == "clean" else 1),
            "Attack_Source_AMC": "None" if key == "clean" else "Original TargetAMC",
            "Moving_SurrogateAMC_Access_During_Attack_Generation": False,
            "Checkpoint_Selection_Uses_Current_Defense_Gradients": False,
        })
    pd.DataFrame(protocol_rows).to_csv(
        log_dir / "method_access_protocol_diversity.csv",
        index=False,
    )

    if args.skip_unified_eval:
        print("[INFO] --skip-unified-eval set; training/resume stage complete.")
        return

    # -------------------------- unified C-Eval -----------------------------
    eval_corpus = common.load_artifact(common.corpus_path("eval"))
    test_df = eval_corpus["splits"]["test"].reset_index(drop=True)
    del eval_corpus
    full_test_samples = len(test_df)
    sample_seed = args.seed + 606
    test_df = shared.stratified_fraction_sample(
        test_df,
        fraction=args.test_sample_fraction,
        seed=sample_seed,
    )
    if args.test_max_samples > 0 and len(test_df) > args.test_max_samples:
        test_df = shared.balanced_limit_by_snr(
            test_df, args.test_max_samples, sample_seed + 1
        )
    realized_fraction = len(test_df) / max(full_test_samples, 1)

    eval_dataset = shared.PhysicalIQDataset(test_df, encoder)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    snr_values = sorted(float(x) for x in np.unique(eval_dataset.snrs))

    # Reload frozen source/attacker/original defense to the evaluation device.
    source_target = freeze_model(source_target.to(device))
    diffusion = freeze_model(diffusion.to(device))
    original_surrogate = freeze_model(original_surrogate.to(device))

    eval_trained: Dict[str, Tuple[nn.Module, dict]] = {}
    for key, ckpt in completed_ckpts.items():
        model = common.SurrogateAMC().to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        freeze_model(model)
        eval_trained[key] = (model, ckpt)

    systems = [
        evaluator.DefenseSystem(
            name="Original-Surrogate",
            family="No-Hardening",
            checkpoint_path=surrogate_checkpoint_path,
            checkpoint=surrogate_ckpt,
            model=original_surrogate,
            normalizer=surrogate_norm,
        )
    ]
    # Stable research-order display.
    for key in [
        "clean",
        "pgd",
        "diffusion_k1",
        "diffusion_multistart_cycle",
        "diffusion_multistep_cycle",
        "diffusion_diverse_cycle",
    ]:
        if key not in eval_trained:
            continue
        model, ckpt = eval_trained[key]
        spec = specs[key]
        systems.append(
            evaluator.DefenseSystem(
                name=spec.display_name,
                family=(
                    "Clean-Finetune-Control" if key == "clean"
                    else "Target-Source-Offline-Hardening"
                ),
                checkpoint_path=output_paths[key],
                checkpoint=ckpt,
                model=model,
                normalizer=surrogate_norm,
            )
        )

    # Evaluation manifest: rerun automatically when model set or evaluation
    # settings change; exact reruns can reuse the completed unified result.
    model_hashes = {
        "Original-Surrogate": surrogate_hash,
    }
    for key in completed_ckpts:
        model_hashes[specs[key].display_name] = sha256_file(output_paths[key])
    eval_signature = {
        "model_hashes": model_hashes,
        "eval_eps_values": [float(x) for x in args.eval_eps_values],
        "classical_attacks": list(args.classical_attacks),
        "sample_fraction": float(args.test_sample_fraction),
        "test_max_samples": int(args.test_max_samples),
        "sample_seed": int(sample_seed),
        "diffusion_hash": diffusion_hash,
        "tanh_temperature": diffusion_temperature,
        "diverse_eval": bool(args.diverse_diffusion_eval),
        "diverse_ddim_steps": [int(x) for x in args.diverse_ddim_steps],
        "diverse_ddim_eta": float(args.diverse_ddim_eta),
    }
    eval_manifest_path = log_dir / "unified_eval_manifest.json"
    global_path = log_dir / "final_global_all_diversity_methods.csv"
    snr_path = log_dir / "final_per_snr_all_diversity_methods.csv"
    diverse_eval_path = log_dir / "diverse_diffusion_transfer_global.csv"

    reuse_eval = False
    if (
        args.auto_resume
        and not args.force_unified_eval
        and eval_manifest_path.exists()
        and global_path.exists()
        and snr_path.exists()
    ):
        try:
            old_sig = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
            reuse_eval = old_sig == eval_signature
        except Exception:
            reuse_eval = False

    if reuse_eval:
        print("[RESUME] Unified evaluation manifest matches; loading existing final CSVs.")
        global_df = pd.read_csv(global_path)
        snr_df = pd.read_csv(snr_path)
        diverse_eval_df = (
            pd.read_csv(diverse_eval_path)
            if diverse_eval_path.exists()
            else pd.DataFrame()
        )
    else:
        print(
            f"[INFO] Unified evaluation: {len(systems)} systems on "
            f"{len(test_df)}/{full_test_samples} held-out samples."
        )
        source_models = {"OriginalTarget": (source_target, target_norm)}
        global_acc = {}
        snr_acc = {}
        evaluator.evaluate_clean(
            loader=eval_loader,
            systems=systems,
            snr_values=snr_values,
            device=device,
            global_acc=global_acc,
            snr_acc=snr_acc,
        )
        diffusion_unit_cache = precompute_diffusion_unit_cache_consistent(
            shared=shared,
            loader=eval_loader,
            attack_diffusion=diffusion,
            attack_normalizer=diffusion_norm,
            attack_betas=diffusion_betas,
            tanh_temperature=diffusion_temperature,
            num_starts=1,
            device=device,
            seed=args.seed + 5_000_000,
        )
        evaluator.evaluate_epsilon_sweep(
            shared=shared,
            loader=eval_loader,
            systems=systems,
            source_models=source_models,
            snr_values=snr_values,
            attack_diffusion=diffusion,
            attack_normalizer=diffusion_norm,
            attack_betas=diffusion_betas,
            eps_values=args.eval_eps_values,
            classical_attacks=args.classical_attacks,
            include_whitebox_hybrid=False,
            include_transfer_hybrid=False,
            hybrid_k=1,
            device=device,
            seed=args.seed,
            keep_ratio=args.keep_ratio,
            global_acc=global_acc,
            snr_acc=snr_acc,
            diffusion_unit_cache=diffusion_unit_cache,
        )
        global_df, snr_df = evaluator.accumulators_to_frames(
            global_acc=global_acc,
            snr_acc=snr_acc,
            sample_fraction=realized_fraction,
            sample_seed=sample_seed,
            full_test_samples=full_test_samples,
        )
        global_df.to_csv(global_path, index=False)
        snr_df.to_csv(snr_path, index=False)

        if args.diverse_diffusion_eval:
            diverse_eval_df = evaluate_diverse_diffusion_transfer_suite(
                shared=shared,
                evaluator=evaluator,
                loader=eval_loader,
                systems=systems,
                attack_diffusion=diffusion,
                attack_normalizer=diffusion_norm,
                attack_betas=diffusion_betas,
                tanh_temperature=diffusion_temperature,
                ddim_steps=args.diverse_ddim_steps,
                ddim_eta=args.diverse_ddim_eta,
                eps_values=args.eval_eps_values,
                device=device,
                seed=args.seed + 7_000_000,
                keep_ratio=args.keep_ratio,
            )
            diverse_eval_df.to_csv(diverse_eval_path, index=False)
        else:
            diverse_eval_df = pd.DataFrame()

        eval_manifest_path.write_text(
            json.dumps(eval_signature, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # --------------------------- comparison tables -------------------------
    pgd_display = specs["pgd"].display_name
    comparison = build_all_vs_pgd_comparison(global_df, pgd_display)
    comparison.to_csv(log_dir / "all_methods_vs_pgd_offline.csv", index=False)
    focus_summary = build_focus_summary(global_df, args.train_eps)
    focus_summary.to_csv(log_dir / "focus_whitebox_summary.csv", index=False)

    run_config = {
        "experiment": "target_source_diffusion_diversity_offline_hardening",
        "seed": args.seed,
        "source_model": "Original TargetAMC",
        "defense_model": "SurrogateAMC",
        "train_eps": args.train_eps,
        "multistart_k": args.multistart_k,
        "ddim_steps": list(args.diverse_ddim_steps),
        "ddim_eta": args.diverse_ddim_eta,
        "requested_experiments": list(args.experiments),
        "completed_experiments": sorted(completed_ckpts.keys()),
        "bank_paths": {k: str(v) for k, v in bank_paths.items()},
        "output_paths": {k: str(v) for k, v in output_paths.items()},
        "auto_resume": args.auto_resume,
        "retrain_all_models": args.retrain_all_models,
        "full_epoch_training": True,
        "configured_epochs": int(args.epochs),
        "early_stopping_enabled": False,
        "cross_method_loss_metric": "Val_Clean_Loss",
        "diversity_signature": diversity_signature,
        "unified_eval_signature": eval_signature,
    }
    (log_dir / "run_config_diversity.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 124)
    print(f" UNIFIED WHITE-BOX SUMMARY AT eps={args.train_eps:g}")
    print(" Lower ASR is better; clean accuracy is shown separately.")
    print("=" * 124)
    print(
        focus_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    if not comparison.empty:
        focus_cmp = comparison[
            comparison["EPS"].notna()
            & np.isclose(
                comparison["EPS"].fillna(-999).to_numpy(float),
                args.train_eps,
            )
            & comparison["Threat_Model"].eq("White-Box")
        ]
        print("\n" + "=" * 124)
        print(" ALL COMPLETED METHODS vs PGD-OFFLINE — CURRENT-DEFENSE WHITE-BOX")
        print(" Positive improvement means the method has lower ASR than PGD-Offline.")
        print("=" * 124)
        print(
            focus_cmp[[
                "Method", "Attack", "EPS", "Method_ASR", "PGDOffline_ASR",
                "ASR_Improvement_pp_MethodMinusPGD",
            ]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    print("\n[SAVE]", log_dir / "attack_bank_diversity.csv")
    print("[SAVE]", dataset_dir / "generation_summary_diversity.csv")
    print("[SAVE]", log_dir / "training_history_diversity.csv")
    print("[SAVE]", log_dir / "loss_history.csv")
    print("[SAVE]", global_path)
    print("[SAVE]", snr_path)
    if args.diverse_diffusion_eval:
        print("[SAVE]", diverse_eval_path)
    print("[SAVE]", log_dir / "all_methods_vs_pgd_offline.csv")
    print("[SAVE]", log_dir / "focus_whitebox_summary.csv")
    print("[SAVE]", log_dir / "run_config_diversity.json")


if __name__ == "__main__":
    main()
