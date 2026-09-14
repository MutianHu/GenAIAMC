"""Stage 3 (0909): train the deployed Target AMC on C-Target-0909 only.

Key 0909 rules:
- load clean_target_0909.pt through pipeline_common_0909.corpus_path("target")
- train/validate/test on the full SNR grid contained in the 0909 target corpus
- save the Target AMC as artifacts/models/target_amc_0909.pt
- save training history as artifacts/logs/target_amc_training_0909.csv
- record source corpus, pipeline version, test accuracy, and training arguments
  in the checkpoint to prevent accidental mixing with the legacy pipeline
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pipeline_common_0909 import (
    GlobalRMSNormalizer,
    IQDataset,
    LOG_ROOT,
    TargetAMC,
    corpus_path,
    evaluate_classifier,
    load_artifact,
    make_label_encoder,
    model_path,
    now_utc,
    pipeline_config,
    save_artifact,
    set_global_seed,
    train_amc,
)


def save_target_checkpoint_0909(
    model: torch.nn.Module,
    normalizer: GlobalRMSNormalizer,
    encoder,
    history: list[dict[str, float]],
    test_accuracy: float,
    source_corpus_path,
    available_snrs: list[float],
    args: argparse.Namespace,
):
    """Save a self-describing 0909 Target AMC checkpoint.

    The checkpoint keeps the fields required by
    pipeline_common_0909.load_amc_checkpoint():
        model_state, normalizer, classes

    Extra metadata is included so downstream diagnostic scripts can verify that
    the Target AMC was trained on the same 0909 receiver-domain data pipeline.
    """
    output = model_path("target")

    payload = {
        "schema_version": 2,
        "created_utc": now_utc(),
        "pipeline_version": "0909",
        "artifact_suffix": "_0909",
        "role": "target",
        "architecture": type(model).__name__,
        "model_state": model.cpu().state_dict(),
        "normalizer": normalizer.state_dict(),
        "classes": encoder.classes_.tolist(),
        "history": history,
        "test_accuracy": float(test_accuracy),
        "source_corpus": "target_0909",
        "source_corpus_path": str(source_corpus_path),
        "training_snr_db": [float(x) for x in available_snrs],
        "training_args": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
        },
        "config": pipeline_config(),
    }

    save_artifact(payload, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the deployed Target AMC from the 0909 C-Target corpus."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2_027)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load only the 0909 Target corpus.
    # corpus_path("target") -> artifacts/data/clean_target_0909.pt
    # ------------------------------------------------------------------
    target_corpus_path = corpus_path("target")
    print(f"[INFO] Loading 0909 Target corpus: {target_corpus_path}")

    corpus = load_artifact(target_corpus_path)

    if str(corpus.get("pipeline_version", "")) != "0909":
        raise RuntimeError(
            f"Expected a 0909 target corpus, but {target_corpus_path} reports "
            f"pipeline_version={corpus.get('pipeline_version')!r}"
        )

    required_splits = {"train", "val", "test"}
    missing_splits = required_splits.difference(corpus.get("splits", {}).keys())
    if missing_splits:
        raise RuntimeError(
            f"0909 target corpus is missing required splits: {sorted(missing_splits)}"
        )

    train_df = corpus["splits"]["train"].reset_index(drop=True)
    val_df = corpus["splits"]["val"].reset_index(drop=True)
    test_df = corpus["splits"]["test"].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            "0909 target corpus contains an empty split: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    available_snrs = sorted(float(x) for x in train_df["SNR_dB"].unique())

    print(f"[INFO] pipeline_version : {corpus.get('pipeline_version')}")
    print(f"[INFO] training SNRs     : {available_snrs}")
    print(f"[INFO] train samples     : {len(train_df)}")
    print(f"[INFO] val samples       : {len(val_df)}")
    print(f"[INFO] test samples      : {len(test_df)}")
    print(f"[INFO] device            : {device}")

    # ------------------------------------------------------------------
    # Target AMC preprocessing/training.
    # The global RMS normalizer is fitted ONLY on the 0909 Target train split.
    # Per-frame AGC has already been applied during 0909 data generation.
    # ------------------------------------------------------------------
    encoder = make_label_encoder()
    normalizer = GlobalRMSNormalizer.fit(train_df)

    model, history = train_amc(
        TargetAMC(),
        train_df,
        val_df,
        encoder,
        normalizer,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # ------------------------------------------------------------------
    # Clean test accuracy on the complete 0909 target test split.
    # ------------------------------------------------------------------
    test_loader = DataLoader(
        IQDataset(test_df, encoder, normalizer),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_accuracy = evaluate_classifier(model, test_loader, device)

    # ------------------------------------------------------------------
    # Save target_amc_0909.pt with explicit 0909 metadata.
    # ------------------------------------------------------------------
    output = save_target_checkpoint_0909(
        model=model,
        normalizer=normalizer,
        encoder=encoder,
        history=history,
        test_accuracy=test_accuracy,
        source_corpus_path=target_corpus_path,
        available_snrs=available_snrs,
        args=args,
    )

    # ------------------------------------------------------------------
    # Save 0909 training log separately from legacy runs.
    # ------------------------------------------------------------------
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    history_path = LOG_ROOT / "target_amc_training_0909.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    print("=" * 72)
    print("0909 Target AMC training finished")
    print(f"Clean-test accuracy : {test_accuracy:.4f}")
    print(f"Checkpoint path     : {output.resolve()}")
    print(f"Checkpoint filename : {output.name}")
    print(f"Training log        : {history_path.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
