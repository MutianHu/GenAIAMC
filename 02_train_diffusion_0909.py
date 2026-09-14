"""
Train clean conditional diffusion model for anomaly detection.

Changes:
- train on the 0909 target AMC clean corpus
- condition only on modulation label
- remove SNR condition
- keep lowest validation-loss checkpoint
- [Update] Add Classifier-Free Guidance (CFG) during training
- [Update] Replace pooling/interpolation with Conv1d(stride=2) and ConvTranspose1d
- [0909] use pipeline_common_0909 / clean_target_0909.pt
- [0909] configurable SNR training grid; default uses the full -10..30 dB grid
- [0909] save checkpoint/log with _0909 suffix
"""

import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pipeline_common_0909 import (
    ARTIFACT_ROOT,
    corpus_path,
    load_artifact,
    save_artifact,
    make_label_encoder,
    GlobalRMSNormalizer,
    dataframe_iq,
    now_utc,
    LOG_ROOT,
    set_global_seed,
)


class DiffusionDataset(Dataset):
    def __init__(self, df, normalizer, encoder):
        self.x = normalizer.transform(dataframe_iq(df))
        self.y = encoder.transform(df["Modulation_Label"].to_numpy())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.x[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.long),
        )


class ResidualBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.c1 = nn.Conv1d(a, b, 3, padding=1)
        self.c2 = nn.Conv1d(b, b, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(b)
        self.bn2 = nn.BatchNorm1d(b)
        self.skip = nn.Identity() if a == b else nn.Conv1d(a, b, 1)

    def forward(self, x):
        h = F.gelu(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return F.gelu(h + self.skip(x))


class LabelOnlyDiffusion(nn.Module):
    """DDPM noise predictor conditioned only by modulation label (with CFG support)."""

    def __init__(self, classes=6, base=64):
        super().__init__()

        self.time = nn.Sequential(
            nn.Linear(1, base*4),
            nn.GELU(),
            nn.Linear(base*4, base*4)
        )

        # [修改] 增加1个额外类别，作为 CFG 中的 "空标签" (Null label)
        self.label = nn.Embedding(classes + 1, base*4)

        self.in_conv = nn.Conv1d(2, base, 3, padding=1)

        # [修改] 使用步长为2的卷积进行下采样 (取代 AvgPool1d)
        self.down1_conv = nn.Conv1d(base, base, kernel_size=4, stride=2, padding=1)
        self.d1 = ResidualBlock(base, base*2)

        self.down2_conv = nn.Conv1d(base*2, base*2, kernel_size=4, stride=2, padding=1)
        self.d2 = ResidualBlock(base*2, base*4)

        self.down3_conv = nn.Conv1d(base*4, base*4, kernel_size=4, stride=2, padding=1)
        self.d3 = ResidualBlock(base*4, base*8)

        self.mid = ResidualBlock(base*8, base*8)

        # [修改] 使用转置卷积进行上采样 (取代 F.interpolate)
        self.up1_conv = nn.ConvTranspose1d(base*8, base*8, kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock(base*12, base*4)

        self.up2_conv = nn.ConvTranspose1d(base*4, base*4, kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock(base*6, base*2)

        self.up3_conv = nn.ConvTranspose1d(base*2, base*2, kernel_size=4, stride=2, padding=1)
        self.up3 = ResidualBlock(base*3, base)

        self.out = nn.Conv1d(base, 2, 3, padding=1)

        self.cond = nn.Linear(base*4, base*8)


    def forward(self, x, t, y):

        c = self.cond(
            self.time(t.float().unsqueeze(1))
            +
            self.label(y)
        ).unsqueeze(-1)

        # [修改] 按照新的上下采样结构重建特征流
        x1 = self.in_conv(x)

        x2 = self.down1_conv(x1)
        x2 = self.d1(x2)

        x3 = self.down2_conv(x2)
        x3 = self.d2(x3)

        x4 = self.down3_conv(x3)
        x4 = self.d3(x4)

        x4 = self.mid(x4 + c)

        x = self.up1_conv(x4)
        x = self.up1(torch.cat([x, x3], 1))

        x = self.up2_conv(x)
        x = self.up2(torch.cat([x, x2], 1))

        x = self.up3_conv(x)
        x = self.up3(torch.cat([x, x1], 1))

        return self.out(x)


def loss_fn(model, x, y, betas, null_label_idx, uncond_prob=0.1):

    ab = torch.cumprod(1-betas, 0)

    t = torch.randint(
        0, len(betas),
        (len(x),),
        device=x.device
    )

    noise = torch.randn_like(x)

    a = ab[t].view(-1,1,1)

    xt = torch.sqrt(a)*x + torch.sqrt(1-a)*noise

    # [修改] CFG 随机掩码机制：以 uncond_prob 概率用空标签替换真实的调制标签
    if uncond_prob > 0.0:
        mask = torch.rand(y.shape, device=y.device) < uncond_prob
        y_cfg = y.clone()
        y_cfg[mask] = null_label_idx
    else:
        y_cfg = y

    pred = model(
        xt,
        t/len(betas),
        y_cfg
    )

    return F.mse_loss(pred, noise)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2040)
    # [新增] CFG 空标签抛弃率参数
    parser.add_argument("--uncond-prob", type=float, default=0.1)
    parser.add_argument(
        "--train-snrs",
        type=float,
        nargs="+",
        default=[-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        help=(
            "SNR values used to train/validate the diffusion detector. "
            "Default: the complete 0909 grid from -10 through 30 dB."
        ),
    )

    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.steps < 1:
        parser.error("epochs, batch-size and steps must all be >= 1")
    if not 0.0 <= args.uncond_prob < 1.0:
        parser.error("--uncond-prob must be in [0,1)")

    set_global_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    target_corpus_path = corpus_path("target")
    print(f"[INFO] Loading 0909 target clean corpus: {target_corpus_path}")

    corpus = load_artifact(target_corpus_path)

    if str(corpus.get("pipeline_version", "")) != "0909":
        raise RuntimeError(
            f"Expected a 0909 corpus, but {target_corpus_path} reports "
            f"pipeline_version={corpus.get('pipeline_version')!r}"
        )

    train_df_all = corpus["splits"]["train"]
    val_df_all = corpus["splits"]["val"]

    requested_snrs = [float(x) for x in args.train_snrs]
    available_train_snrs = sorted(float(x) for x in train_df_all["SNR_dB"].unique())
    missing_snrs = [
        snr for snr in requested_snrs
        if not any(np.isclose(snr, available) for available in available_train_snrs)
    ]
    if missing_snrs:
        raise ValueError(
            f"Requested training SNR values are absent from the 0909 target corpus: "
            f"{missing_snrs}; available={available_train_snrs}"
        )

    train_mask = np.zeros(len(train_df_all), dtype=bool)
    val_mask = np.zeros(len(val_df_all), dtype=bool)
    for snr in requested_snrs:
        train_mask |= np.isclose(train_df_all["SNR_dB"].to_numpy(dtype=float), snr)
        val_mask |= np.isclose(val_df_all["SNR_dB"].to_numpy(dtype=float), snr)

    train_df = train_df_all.loc[train_mask].reset_index(drop=True)
    val_df = val_df_all.loc[val_mask].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise RuntimeError(
            f"0909 diffusion train/val split became empty after SNR filtering: "
            f"train={len(train_df)}, val={len(val_df)}"
        )

    print(f"[INFO] 0909 corpus pipeline_version : {corpus.get('pipeline_version')}")
    print(f"[INFO] Available corpus SNRs       : {available_train_snrs}")
    print(f"[INFO] Diffusion training SNRs     : {requested_snrs}")
    print(f"[INFO] Train samples               : {len(train_df)} / {len(train_df_all)}")
    print(f"[INFO] Validation samples          : {len(val_df)} / {len(val_df_all)}")

    encoder = make_label_encoder()

    # [新增] 计算 CFG 空标签的索引 (例如原有6个类0-5，空标签索引即为6)
    num_classes = len(encoder.classes_)
    null_label_idx = num_classes

    normalizer = GlobalRMSNormalizer.fit(
        train_df
    )


    train_shuffle_generator = torch.Generator()
    train_shuffle_generator.manual_seed(args.seed + 101)
    train_loader = DataLoader(
        DiffusionDataset(train_df, normalizer, encoder),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_shuffle_generator,
    )

    val_loader = DataLoader(
        DiffusionDataset(val_df, normalizer, encoder),
        batch_size=args.batch_size
    )


    model = LabelOnlyDiffusion(
        classes=num_classes
    ).to(device)


    betas = torch.linspace(
        1e-4,
        0.02,
        args.steps,
        device=device
    )


    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr
    )


    best_loss = float("inf")
    best_state = None
    history=[]


    for epoch in range(1,args.epochs+1):

        model.train()
        losses=[]

        for x,y in tqdm(train_loader):

            x,y=x.to(device),y.to(device)

            # 训练阶段：引入 CFG，一定概率丢弃标签
            loss=loss_fn(
                model, x, y, betas,
                null_label_idx=null_label_idx,
                uncond_prob=args.uncond_prob
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),1.0
            )
            opt.step()

            losses.append(loss.item())


        model.eval()
        vals=[]

        with torch.no_grad():
            for x,y in val_loader:
                # 验证阶段：关闭随机掩码 (uncond_prob=0.0)，为了评估最稳定的条件损失
                vals.append(
                    loss_fn(
                        model,
                        x.to(device),
                        y.to(device),
                        betas,
                        null_label_idx=null_label_idx,
                        uncond_prob=0.0
                    ).item()
                )


        train_loss=float(np.mean(losses))
        val_loss=float(np.mean(vals))

        history.append(
            {
                "epoch":epoch,
                "train_loss":train_loss,
                "val_loss":val_loss
            }
        )

        print(
            f"epoch={epoch:03d} "
            f"train={train_loss:.6f} "
            f"val={val_loss:.6f}"
        )

        if val_loss < best_loss:
            best_loss=val_loss
            best_state=copy.deepcopy(
                model.state_dict()
            )
            print("[INFO] best checkpoint updated")


    model.load_state_dict(best_state)


    output = (
        ARTIFACT_ROOT
        / "models/clean_diffusion_target_label_only_0909.pt"
    )


    save_artifact(
        {
            "schema_version":1,
            "created_utc":now_utc(),
            "architecture":
                "LabelOnlyDiffusion",
            "model_state":
                model.cpu().state_dict(),
            "normalizer":
                normalizer.state_dict(),
            "classes":
                encoder.classes_.tolist(),
            "condition":
                "modulation_only",
            "source_corpus":
                "target_0909",
            "source_corpus_path":
                str(target_corpus_path),
            "pipeline_version":
                "0909",
            "training_snr_db":
                requested_snrs,
            "ddpm_steps":
                args.steps,
            "seed":
                args.seed,
            "best_val_loss":
                best_loss,
            # [新增] 记录 CFG 配置，以供下游推理阶段使用
            "cfg_enabled": True,
            "cfg_uncond_prob": args.uncond_prob,
            "cfg_null_label_index": null_label_idx,
            "history":
                history
        },
        output
    )


    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.DataFrame(history).to_csv(
        LOG_ROOT /
        "clean_diffusion_target_label_only_0909.csv",
        index=False
    )


    print("="*60)
    print("Diffusion training finished")
    print(f"Best validation loss: {best_loss:.8f}")
    print(f"Checkpoint path: {output.resolve()}")
    print(f"Checkpoint filename: {output.name}")
    print("="*60)


if __name__ == "__main__":
    main()
