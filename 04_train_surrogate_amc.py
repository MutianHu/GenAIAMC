"""Stage 4: train the attacker's architecture-distinct surrogate AMC on C-Surrogate."""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pipeline_common_0909 import (
    GlobalRMSNormalizer,
    IQDataset,
    LOG_ROOT,
    SurrogateAMC,
    corpus_path,
    evaluate_classifier,
    load_artifact,
    make_label_encoder,
    save_amc_checkpoint,
    set_global_seed,
    train_amc,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the transfer-attacker surrogate AMC from C-Surrogate.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    print("[INFO] 正在初始化环境与全局随机种子...")
    set_global_seed(2_028)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] 正在从本地磁盘加载 C-Surrogate 数据集 (若数据量大可能需要几十秒)...")
    corpus = load_artifact(corpus_path("surrogate"))

    print("[INFO] 数据集加载完成！正在计算训练集的全局 RMS 归一化系数并拟合标签...")
    encoder = make_label_encoder()
    normalizer = GlobalRMSNormalizer.fit(corpus["splits"]["train"])

    print(f"[INFO] 数据预处理完毕。开始在 {device} 上训练 SurrogateAMC 模型 (每个 Epoch 结束后会有输出)...")
    model, history = train_amc(
        SurrogateAMC(), corpus["splits"]["train"], corpus["splits"]["val"], encoder, normalizer, device,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
    )

    print("[INFO] 训练循环结束！正在构建测试集 DataLoader (DataFrame 到 Tensor 的转换可能会耗时)...")
    test_loader = DataLoader(IQDataset(corpus["splits"]["test"], encoder, normalizer), args.batch_size)

    print("[INFO] 正在执行测试集推理评估...")
    test_accuracy = evaluate_classifier(model, test_loader, device)

    print("[INFO] 评估完毕。正在保存模型权重 (Checkpoints) 与训练日志...")
    output = save_amc_checkpoint("surrogate", model, normalizer, encoder, history)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(LOG_ROOT / "surrogate_amc_training_0909.csv", index=False)

    print(f"surrogate clean-test accuracy={test_accuracy:.4f}; saved {output}")


if __name__ == "__main__":
    main()