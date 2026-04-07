# ====================================================================
# SCRIPT 3: AMC 训练与 G-EAMD 评估 (amc_eval_stage.py)
# [最终版 - 已修复所有 NameError Bug]
# [!! G-EAMD v2 !!] 切换到 "条件损失" 异常分数
# [!! G-EAMD v3 !!] 同时报告 TPR 和 FPR
# ====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
# [删除] 不再需要 train_test_split
# from sklearn.model_selection import train_test_split
import re
from typing import Dict
import os
import ast  # 新增：用于安全解析数据
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix

# --- [核心修改] 新的数据集路径 ---
AMC_TRAINING_LOG_PATH = 'amc_enhanced_training_log.csv'
CONDITIONAL_DDPM_MODEL_PATH = 'conditional_dfm_unet_model.pth'  # 条件DDPM模型路径
CONDITIONAL_SYNTHETIC_DATA_PATH = 'conditional_synthetic_data.csv'  # 条件合成数据路径
# AMC_BASE_PATH = 'amc_enhanced_model.pth'
AMC_BASE_PATH = 'amc_base_model.pth'
AMC_ENHANCED_PATH = 'amc_enhanced_model.pth'

# [新增] 预划分的数据文件路径
CLEAN_TRAIN_PATH = 'clean_train.csv'
CLEAN_VAL_PATH = 'clean_val.csv'
CLEAN_TEST_PATH = 'clean_test.csv'
PGD_TEST_PATH = 'pgd_malicious_data.csv'
# --- 结束修改 ---

# --- 全局配置 ---
SAMPLES_PER_SIGNAL = 256
CHANNELS = 2
T = 500  # 扩散步数

# --- [新增] CFG (Classifier-Free Guidance) 配置 ---
# 必须与 SCRIPT 2 (训练脚本) 严格一致
MODULATION_TYPES = ['GFSK', 'QPSK', '16QAM']
NUM_MODULATION_CLASSES = len(MODULATION_TYPES)
NULL_LABEL_INDEX = NUM_MODULATION_CLASSES  # "空标签"索引 (例如: 3)
NUM_EMBEDDING_CLASSES = NUM_MODULATION_CLASSES + 1  # 总嵌入类别数 (例如: 4)


# --- 结束新增 ---


# --- [核心修复] 添加完整的 DDPM 参数定义 (与 SCRIPT 2 保持一致) ---
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


# 定义DDPM参数
betas = linear_beta_schedule(T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 将参数移动到GPU（如果可用）
# [修改] 我们将在主函数中定义 device, 但在这里先把参数移到 CPU
# 这样在导入时不会出错，然后在主函数中再移到 GPU
# 或者, 我们在这里就定义全局 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
alphas_cumprod_prev = alphas_cumprod_prev.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)


# --- 结束修复 ---


# --- [核心修复] 插入缺失的 extract 函数 ---
def extract(a, t, x_shape):
    b = t.shape[0]
    # 确保 gather 在正确的设备上
    out = a.to(t.device).gather(-1, t.to(t.device))
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# --- 结束修复 ---


# --- 模型结构定义 ---
class ConvBlock(nn.Module):
    """用于 DDPM UNet 的标准 1D 卷积块 (非残差)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(h)))


# --- 更新：条件UNet1D模型定义 ---
class ConditionalUNet1D(nn.Module):
    """
    条件生成扩散模型 - 添加调制标签条件
    [注意] 此定义与 SCRIPT 2 中不同，它接受 num_modulation_classes 参数。
    我们将传入 NUM_EMBEDDING_CLASSES。
    """

    def __init__(self, in_channels=CHANNELS, out_channels=CHANNELS, base_channels=32, num_conv_blocks=2,
                 num_modulation_classes=4):  # <-- 接受参数
        super().__init__()
        self.num_conv_blocks = num_conv_blocks

        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_channels * 4), nn.GELU(), nn.Linear(base_channels * 4, base_channels * 4)
        )

        # 新增：调制标签嵌入 (使用传入的 num_modulation_classes)
        self.modulation_embed = nn.Sequential(
            nn.Embedding(num_modulation_classes, base_channels * 4),  # <-- 使用参数
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )

        # 合并时间和调制嵌入
        self.condition_merge = nn.Linear(base_channels * 8, base_channels * 4)

        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        down_channels = [base_channels * (2 ** i) for i in range(3)]
        up_channels = down_channels[::-1]

        def make_conv_sequence(in_c, out_c, num_blocks):
            blocks = []
            blocks.append(ConvBlock(in_c, out_c))
            for _ in range(num_blocks - 1):
                blocks.append(ConvBlock(out_c, out_c))
            return nn.Sequential(*blocks)

        self.down_blocks = nn.ModuleList([
            make_conv_sequence(down_channels[0], down_channels[0], self.num_conv_blocks),
            nn.Conv1d(down_channels[0], down_channels[1], kernel_size=4, stride=2, padding=1),
            make_conv_sequence(down_channels[1], down_channels[1], self.num_conv_blocks),
            nn.Conv1d(down_channels[1], down_channels[2], kernel_size=4, stride=2, padding=1),
        ])

        self.bottleneck = nn.Sequential(
            ConvBlock(down_channels[2], down_channels[2]),
            ConvBlock(down_channels[2], down_channels[2])
        )

        self.up_blocks = nn.ModuleList([
            nn.ConvTranspose1d(up_channels[0], up_channels[1], kernel_size=4, stride=2, padding=1),
            make_conv_sequence(up_channels[0], up_channels[1], self.num_conv_blocks),
            nn.ConvTranspose1d(up_channels[1], up_channels[2], kernel_size=4, stride=2, padding=1),
            make_conv_sequence(up_channels[1], up_channels[2], self.num_conv_blocks),
        ])

        self.conv_out = nn.Conv1d(up_channels[2], out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, modulation_labels):
        # 时间嵌入
        t_emb = self.time_embed(t.unsqueeze(1))

        # 调制标签嵌入
        mod_emb = self.modulation_embed(modulation_labels)

        # 合并条件
        condition = torch.cat([t_emb, mod_emb], dim=1)
        condition = self.condition_merge(condition)

        # 将条件信息添加到输入中
        batch_size, _, seq_len = x.shape
        condition_expanded = condition.unsqueeze(-1).expand(-1, -1, seq_len)

        x = self.conv_in(x)
        # 将条件信息与输入特征结合
        x = x + condition_expanded[:, :x.shape[1], :]

        skip_connections = [x]

        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i % 2 == 0:
                skip_connections.append(x)

        x = self.bottleneck(x)

        for i, block in enumerate(self.up_blocks):
            if i % 2 == 0:
                x = block(x)
            else:
                skip = skip_connections.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x)

        x = self.conv_out(x)
        return x


class AMCClassifier(nn.Module):
    """基于 1D CNN 的调制分类器 (使用 ResidualBlock)"""

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.norm2 = nn.BatchNorm1d(out_channels)

            # 残差连接
            if in_channels != out_channels or stride != 1:
                self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            else:
                self.residual_conv = nn.Identity()

        def forward(self, x):
            h = F.relu(self.norm1(self.conv1(x)))
            h = self.norm2(self.conv2(h))
            return F.relu(h + self.residual_conv(x))

    def __init__(self, input_samples=SAMPLES_PER_SIGNAL, in_channels=CHANNELS, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # 使用步长为2的残差块替代第一个最大池化层
            self.ResidualBlock(64, 64, stride=2),
            # self.ResidualBlock(64, 64),
            # 使用步长为2的残差块替代第二个最大池化层
            self.ResidualBlock(64, 64, stride=2),
            # self.ResidualBlock(64, 64),
            # 使用步长为2的残差块替代第三个最大池化层
            self.ResidualBlock(64, 64, stride=2),
            # self.ResidualBlock(64, 64),
            self.ResidualBlock(64, 64, stride=2),
            self.ResidualBlock(64, 64)  # 最后一个块保持stride=1
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).squeeze(-1)
        return self.classifier(x)


# --- 改进的数据集类 ---
class SignalDataset(Dataset):
    def __init__(self, df, label_encoder=None, phase='train'):
        # 安全地处理I/Q数据
        I_data = []
        Q_data = []
        labels_str = []

        # [新增] 为 AMC 准确率测试准备 SNR 数据
        self.snrs = []
        self.has_snr = 'SNR_dB' in df.columns

        for idx, row in df.iterrows():
            # 处理I路数据
            i_val = row['I']
            if isinstance(i_val, str):
                try:
                    # [修改] 使用 ast.literal_eval 来安全地解析列表字符串
                    i_array = ast.literal_eval(i_val)
                except:
                    i_array = np.zeros(SAMPLES_PER_SIGNAL)
            else:
                i_array = i_val

            # 处理Q路数据
            q_val = row['Q']
            if isinstance(q_val, str):
                try:
                    # [修改] 使用 ast.literal_eval 来安全地解析列表字符串
                    q_array = ast.literal_eval(q_val)
                except:
                    q_array = np.zeros(SAMPLES_PER_SIGNAL)
            else:
                q_array = q_val

            # 确保数据长度正确
            i_array = np.array(i_array, dtype=np.float32)
            q_array = np.array(q_array, dtype=np.float32)

            if len(i_array) != SAMPLES_PER_SIGNAL:
                i_array = np.resize(i_array, SAMPLES_PER_SIGNAL)
            if len(q_array) != SAMPLES_PER_SIGNAL:
                q_array = np.resize(q_array, SAMPLES_PER_SIGNAL)

            I_data.append(i_array)
            Q_data.append(q_array)
            labels_str.append(row['Modulation_Label'])

            if self.has_snr:
                self.snrs.append(row['SNR_dB'])

        # 堆叠数据
        I_data = np.stack(I_data)
        Q_data = np.stack(Q_data)
        self.data = np.stack([I_data, Q_data], axis=1)
        self.labels_str = np.array(labels_str)
        if self.has_snr:
            self.snrs = np.array(self.snrs)

        if phase == 'train' and label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels_int = self.label_encoder.fit_transform(self.labels_str)
        elif label_encoder is not None:
            self.label_encoder = label_encoder
            # 处理未知标签
            known_labels = [l for l in self.labels_str if l in self.label_encoder.classes_]
            if len(known_labels) != len(self.labels_str):
                # 找出不在 encoder.classes_ 中的标签
                unknown_labels = set(self.labels_str) - set(self.label_encoder.classes_)
                if len(unknown_labels) > 0:
                    print(f"警告: 数据集(phase={phase})发现未知标签: {unknown_labels}。将它们映射到第一个已知标签。")

                temp_labels = np.array(
                    [l if l in self.label_encoder.classes_ else self.label_encoder.classes_[0]
                     for l in self.labels_str])
                self.labels_int = self.label_encoder.transform(temp_labels)
            else:
                self.labels_int = self.label_encoder.transform(self.labels_str)
        else:
            self.labels_int = np.zeros(len(self.data))

        print(f"数据集(phase={phase})加载完成: {len(self.data)} 个样本, 形状: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # [修改] 同时返回 SNR (如果存在)
        item = (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels_int[idx], dtype=torch.long)
        )
        if self.has_snr:
            return item + (self.snrs[idx],)
        return item


# --- [核心修改] train_amc_classifier (现在接收 df_val) ---
def train_amc_classifier(df_train, df_val, label_encoder, model, epochs, batch_size, lr, device, save_path):
    """
    AMC 分类器监督训练函数
    [修改] 接收一个独立的 df_val 用于验证和保存最佳模型
    """
    df_train_legit = df_train[df_train['Anomaly_Label'] == 'Legit'].copy()

    # [新增] 处理 df_val
    df_val_legit = df_val[df_val['Anomaly_Label'] == 'Legit'].copy()

    # 处理合成数据的标签
    synthetic_indices = df_train_legit[
                            'Data_Source'] == 'Synthetic' if 'Data_Source' in df_train_legit.columns else pd.Series(
        [False] * len(df_train_legit))
    if synthetic_indices.any():
        print(f"检测到 {synthetic_indices.sum()} 个合成样本，使用其原有调制标签")

    train_dataset = SignalDataset(df_train_legit, label_encoder=label_encoder, phase='train')
    val_dataset = SignalDataset(df_val_legit, label_encoder=label_encoder, phase='val')  # [新增]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)  # [新增]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 初始化训练日志
    training_log = []
    start_time = datetime.now()
    best_val_accuracy = 0.0  # [新增]
    best_model_state = None  # [新增]
    best_epoch = 0  # [新增]
    patience = 100  # [新增] 早停耐心
    patience_counter = 0  # [新增]

    print(f"开始训练 AMC 模型, 训练数据量: {len(train_dataset)} 样本, 验证数据量: {len(val_dataset)} 样本")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []  # 记录每个batch的loss
        total_loss = 0
        corrects = 0

        for batch in train_dataloader:
            data, target = batch[0].to(device), batch[1].to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            total_loss += batch_loss
            preds = output.argmax(dim=1)
            corrects += (preds == target).sum().item()

        # 计算训练集指标
        avg_epoch_loss = total_loss / len(train_dataloader)
        epoch_std_loss = np.std(epoch_losses)
        train_acc = corrects / len(train_dataset)

        # --- [新增] 验证循环 ---
        model.eval()
        val_total_loss = 0
        val_corrects = 0
        with torch.no_grad():
            for batch in val_dataloader:
                data, target = batch[0].to(device), batch[1].to(device)
                output = model(data)
                loss = criterion(output, target)
                val_total_loss += loss.item()
                preds = output.argmax(dim=1)
                val_corrects += (preds == target).sum().item()

        val_avg_loss = val_total_loss / len(val_dataloader)
        val_acc = val_corrects / len(val_dataset)
        # --- 结束新增 ---

        # 记录训练日志
        log_entry = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': avg_epoch_loss,
            'train_loss_std': epoch_std_loss,
            'train_accuracy': train_acc,
            'val_loss': val_avg_loss,  # [新增]
            'val_accuracy': val_acc,  # [新增]
            'learning_rate': lr,
            'batch_size': batch_size
        }
        training_log.append(log_entry)

        # 实时保存训练日志到CSV
        log_df = pd.DataFrame(training_log)
        log_df.to_csv(AMC_TRAINING_LOG_PATH, index=False)

        # --- [新增] 检查最佳模型 ---
        best_marker = ""
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
            best_marker = " * (新最佳模型)"
        else:
            patience_counter += 1
        # --- 结束新增 ---

        # 打印训练进度（每个epoch都打印）
        print(f"Epoch {epoch + 1:3d}/{epochs} | "
              f"Train Loss: {avg_epoch_loss:.4f} (Acc: {train_acc:.4f}) | "
              f"Val Loss: {val_avg_loss:.4f} (Acc: {val_acc:.4f}){best_marker}")

        # [新增] 早停
        if patience_counter >= patience:
            print(f"\n早停触发: 验证集准确率在 {patience} 个epoch内未提升。")
            break

    # 训练完成后的总结
    end_time = datetime.now()
    training_duration = end_time - start_time

    print("\n" + "=" * 80)
    print("AMC 模型训练完成!")
    print(f"训练总时长: {training_duration}")
    print(f"训练日志已保存至: {AMC_TRAINING_LOG_PATH}")

    # [新增] 加载最佳模型
    if best_model_state:
        print(f"加载 Epoch {best_epoch} 的最佳模型 (验证集准确率: {best_val_accuracy:.4f})")
        model.load_state_dict(best_model_state)
    else:
        print("警告: 未能保存任何最佳模型，将使用最后一个 epoch 的模型。")

    torch.save(model.state_dict(), save_path)
    print(f"AMC 模型权重已保存至: {save_path}")

    return model, training_log


# --- 结束修改 ---


# --- [核心新增] 轻量级 AMC 准确率测试函数 ---
@torch.no_grad()
def test_amc_accuracy(model, df_test, encoder, device, title="AMC 准确率测试"):
    """
    [!! 修改 !!] 评估 AMC 模型的准确率 (包括按 SNR)。
    不涉及 DFM。
    """
    print(f"\n{title}")
    print("-" * 50)

    model.eval()
    # [核心 Bug 修复] 此处使用 'encoder' 而不是 'label_encoder'
    # [!! 修改 !!] 传入 df_test
    dataset = SignalDataset(df_test, encoder, phase='test')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    corrects = 0
    total = 0

    # 用于按 SNR 统计
    snr_corrects = {}
    snr_totals = {}

    for batch in tqdm(dataloader, desc=f"  > {title}"):
        data, target = batch[0].to(device), batch[1].to(device)
        snrs = batch[2].numpy()  # [新增] 获取 SNR

        output = model(data)
        preds = output.argmax(dim=1)

        corrects_batch = (preds == target).sum().item()
        corrects += corrects_batch
        total += target.size(0)

        # [新增] 按 SNR 累加
        for i in range(len(snrs)):
            snr = snrs[i]
            if snr not in snr_totals:
                snr_totals[snr] = 0
                snr_corrects[snr] = 0
            snr_totals[snr] += 1
            if (preds[i] == target[i]):
                snr_corrects[snr] += 1

    overall_accuracy = (corrects / total) * 100
    print(f"总体分类精度: {overall_accuracy:.2f}% ({corrects}/{total})")

    accuracy_by_snr = {}
    print("按SNR分类精度:")
    for snr in sorted(snr_totals.keys()):
        snr_acc = (snr_corrects[snr] / snr_totals[snr]) * 100
        accuracy_by_snr[snr] = snr_acc
        print(f"  SNR {snr}dB: {snr_acc:.2f}% ({snr_corrects[snr]}/{snr_totals[snr]})")

    return {
        'Average_Accuracy': overall_accuracy,
        'SNR_Accuracy': accuracy_by_snr
    }


# --- 结束新增 ---


# --- [!! G-EAMD v2 !!] 修改 G-EAMD 评估函数 (使用 DFM 条件损失) ---
@torch.no_grad()
def evaluate_geamd(df_test, dfm_model, amc_model, label_encoder, device='cuda'):
    """
    G-EAMD 评估函数
    [!! 核心修改 v2 !!]
    使用 (AMC预测的标签) 作为 (DFM的条件)，
    并使用 DFM 的 (噪声预测MSE损失) 作为 (异常分数)。
    我们假设 恶意信号 会导致 (信号) 和 (AMC的错误预测) 之间不匹配，
    从而导致 DFM 损失 (异常分数) 显著 升高。
    """
    print("正在进行 G-EAMD v2 评估 (使用 DFM 条件损失 作为异常分数)...")  # [!! 修改 !!]

    if dfm_model is None:
        print("警告: 无扩散模型，跳过 G-EAMD 评估")
        return None, {}, {}

    dfm_model.eval()
    amc_model.eval()

    dataset = SignalDataset(df_test, label_encoder=label_encoder, phase='eval')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    results = []
    legit_scores_by_snr = {}
    anomaly_scores_by_snr = {}

    # [!! 修改 !!] 尝试一个较小但不太小的时间步
    fixed_t = T // 10  # 尝试 t=50 (T=500)
    print(f"  > 使用固定时间步 t={fixed_t} 进行异常分数计算")

    # [修改] 调整 TQDM 描述
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="G-EAMD v2 异常分数计算")):  # [!! 修改 !!]
        data, true_labels = batch[0].to(device), batch[1].to(device)
        snrs = batch[2].numpy()  # 获取 SNR

        # 1. AMC预测 (c_pred)
        with torch.no_grad():
            amc_output = amc_model(data)
            amc_pred = amc_output.argmax(dim=1)
            amc_confidence = F.softmax(amc_output, dim=1).max(dim=1)[0]

        # 2. 准备 DFM 输入 (x_t 和 真实噪声 ε)
        batch_size = data.shape[0]
        t = torch.full((batch_size,), fixed_t, device=device, dtype=torch.long)
        noise = torch.randn_like(data)  # 真实噪声 ε

        sqrt_alpha_bar_t = extract(sqrt_alphas_cumprod, t, data.shape)
        sqrt_one_minus_alpha_bar_t = extract(sqrt_one_minus_alphas_cumprod, t, data.shape)

        x_t = sqrt_alpha_bar_t * data + sqrt_one_minus_alpha_bar_t * noise
        time_input = t.float() / T * 999.0

        # --- [!! 核心修改 !!] ---
        # 3. 使用 AMC 的预测 (amc_pred) 作为 DFM 的条件
        noise_pred = dfm_model(x_t, time_input, amc_pred)

        # 4. 计算 DFM 的噪声预测损失 (MSE) 作为新的 "异常分数"
        #    我们计算每个样本的损失，而不是整个批次的平均值
        loss_per_sample = F.mse_loss(noise, noise_pred, reduction='none')

        #    (B, C, L) -> (B)
        #    anomaly_score 是每个样本的 (均方) 误差
        anomaly_score = loss_per_sample.mean(dim=[1, 2])
        # --- [!! 结束修改 !!] ---

        # 5. 收集结果 (这部分不变)
        for i in range(batch_size):
            true_label_str = label_encoder.inverse_transform([true_labels[i].cpu().numpy()])[0]
            pred_label_str = label_encoder.inverse_transform([amc_pred[i].cpu().numpy()])[0]

            original_idx = (batch_idx * 128) + i
            if original_idx >= len(df_test): continue

            original_row = df_test.iloc[original_idx]
            anomaly_label = original_row['Anomaly_Label']
            snr_val = original_row['SNR_dB']

            result = {
                'Index': original_row.name,
                'True_Label': true_label_str,
                'AMC_Pred': pred_label_str,
                'AMC_Confidence': amc_confidence[i].item(),
                'Anomaly_Score': anomaly_score[i].item(),  # [!!] 这是新的分数
                'SNR_dB': snr_val,
                'True_Anomaly': anomaly_label != 'Legit',
                'Anomaly_Label': anomaly_label
            }
            results.append(result)

            if anomaly_label == 'Legit':
                if snr_val not in legit_scores_by_snr:
                    legit_scores_by_snr[snr_val] = []
                legit_scores_by_snr[snr_val].append(anomaly_score[i].item())
            else:
                if snr_val not in anomaly_scores_by_snr:
                    anomaly_scores_by_snr[snr_val] = []
                anomaly_scores_by_snr[snr_val].append(anomaly_score[i].item())

    # (这部分不变)
    legit_avg_scores = {snr: np.mean(errors) for snr, errors in legit_scores_by_snr.items()}
    anomaly_avg_scores = {snr: np.mean(errors) for snr, errors in anomaly_scores_by_snr.items()}

    results_df = pd.DataFrame(results)
    return results_df, legit_avg_scores, anomaly_avg_scores


# --- [!! G-EAMD v3 !!] DFM 异常检测率评估函数 (按 SNR 划分, 报告 FPR) ---
@torch.no_grad()
def evaluate_anomaly_detection_rate(df_results, title="DFM 异常检测性能报告"):
    """
    [!! 核心修改 v3 !!]
    1. 按 SNR 分组计算阈值和检测率。
    2. 我们假设 异常分数 (高) > 合法分数 (低)。
    3. [新增] 同时报告误报率 (FPR)。
    """
    if df_results is None or len(df_results) == 0:
        print(f"\n{title}: 无数据可评估")
        return None

    print(f"\n{title}")
    print("-" * 50)

    # 按 SNR 分组
    all_snrs = sorted(df_results['SNR_dB'].unique())

    total_anomalies = 0
    total_detected = 0
    total_legit = 0  # [!! 新增 !!]
    total_false_positives = 0  # [!! 新增 !!]

    # [!! 修改 !!] 调整阈值的保守程度 (sigma_multiplier)
    # 3.0 = 非常保守 (低 FPR, 低 TPR)
    # 1.5 = 比较激进 (高 FPR, 高 TPR)
    SIGMA_MULTIPLIER = 1.5
    print(f"!!! 使用的检测阈值策略: Mean + {SIGMA_MULTIPLIER}*Std !!!")

    print(
        f"{'SNR (dB)':<10} | {f'阈值 (μ+{SIGMA_MULTIPLIER}σ)':<15} | {'检测率 (TPR)':<15} | {'误报率 (FPR)':<15} | {'TP/TotalMal':<15} | {'FP/TotalLegit':<15}")  # [!! 修改 !!]
    print("-" * 90)  # [!! 修改 !!]

    for snr in all_snrs:
        df_snr = df_results[df_results['SNR_dB'] == snr]

        # --- 1. 阈值校准 (Calibration) ---
        legit_errors = df_snr[df_snr['True_Anomaly'] == False]['Anomaly_Score']

        # [!! 新增 !!] FPR 计算
        num_legit_in_group = 0
        false_positives_in_group = 0
        fpr = 0.0

        if len(legit_errors) > 0:
            threshold_mean = legit_errors.mean()
            threshold_std = legit_errors.std()
            threshold = threshold_mean + SIGMA_MULTIPLIER * threshold_std

            # [!! 新增 !!]
            num_legit_in_group = len(legit_errors)
            false_positives_in_group = (legit_errors > threshold).sum()
            fpr = false_positives_in_group / num_legit_in_group if num_legit_in_group > 0 else 0.0
            total_legit += num_legit_in_group
            total_false_positives += false_positives_in_group
        else:
            threshold = 999.0  # 如果这个SNR没有合法数据（不应发生），则使用一个回退值

        # --- 2. 评估检测率 (仅非法数据) ---
        anomaly_results = df_snr[df_snr['True_Anomaly'] == True]

        num_anomalies_in_group = 0
        detected_anomalies_in_group = 0
        detection_rate = 0.0

        if len(anomaly_results) > 0:
            y_scores_anomaly = anomaly_results['Anomaly_Score'].values
            num_anomalies_in_group = len(y_scores_anomaly)

            y_pred = y_scores_anomaly > threshold

            detected_anomalies_in_group = y_pred.sum()

            detection_rate = detected_anomalies_in_group / num_anomalies_in_group if num_anomalies_in_group > 0 else 0.0

            total_anomalies += num_anomalies_in_group
            total_detected += detected_anomalies_in_group

            print(
                f"{snr:<10} | {threshold:<15.6f} | {detection_rate:<15.4f} | {fpr:<15.4f} | {f'({detected_anomalies_in_group}/{num_anomalies_in_group})':<15} | {f'({false_positives_in_group}/{num_legit_in_group})':<15}")  # [!! 修改 !!]
        else:
            print(
                f"{snr:<10} | {threshold:<15.6f} | {'N/A (无恶意数据)':<15} | {fpr:<15.4f} | {'(0/0)':<15} | {f'({false_positives_in_group}/{num_legit_in_group})':<15}")  # [!! 修改 !!]

    # --- 3. 报告总体结果 ---
    overall_detection_rate = total_detected / total_anomalies if total_anomalies > 0 else 0.0
    overall_fpr = total_false_positives / total_legit if total_legit > 0 else 0.0  # [!! 新增 !!]

    print("-" * 90)  # [!! 修改 !!]
    print(f"\n总体非法信号检测率 (Recall / TPR): {overall_detection_rate:.4f}")
    print(f"  > 含义: {overall_detection_rate * 100:.2f}% 的非法信号被成功检测。")
    print(f"  > 原始数据: 成功检测到 {total_detected} 个 (共 {total_anomalies} 个非法信号)")

    # [!! 新增 !!]
    print(f"\n总体合法信号误报率 (FPR): {overall_fpr:.4f}")
    print(f"  > 含义: {overall_fpr * 100:.2f}% 的合法信号被错误检测。")
    print(f"  > 原始数据: 错误检测到 {total_false_positives} 个 (共 {total_legit} 个合法信号)")

    return {
        'detection_rate': overall_detection_rate,
        'false_positive_rate': overall_fpr
    }


# --- 结束修改 ---


# --- [核心修改] 重写主函数 (run_amc_eval_stage) ---
def run_amc_eval_stage():
    # [修改] 将 device 定义移到全局，以便 DDPM 参数可以访问
    # 这个 device 变量在脚本顶部已被定义
    print(f"使用设备: {device}")

    # --- 1. 加载所有*预划分*的数据 ---
    try:
        print("正在加载预划分的数据集...")
        df_train_clean = pd.read_csv(CLEAN_TRAIN_PATH)
        print(f"  > 成功加载合法训练集: {len(df_train_clean)} 个样本")

        df_val_clean = pd.read_csv(CLEAN_VAL_PATH)
        print(f"  > 成功加载合法验证集: {len(df_val_clean)} 个样本")

        df_test_clean = pd.read_csv(CLEAN_TEST_PATH)
        print(f"  > 成功加载合法测试集: {len(df_test_clean)} 个样本")

        df_test_pgd = pd.read_csv(PGD_TEST_PATH)
        print(f"  > 成功加载恶意测试集: {len(df_test_pgd)} 个样本")

        print("正在加载合成数据 (用于增强)...")
        df_synthetic = pd.read_csv(CONDITIONAL_SYNTHETIC_DATA_PATH)
        print(f"  > 成功加载条件合成数据: {len(df_synthetic)} 个样本")

    except FileNotFoundError as e:
        print(f"错误: 未找到数据文件 {e}")
        print("请确保您已经先运行了 SCRIPT 1 (数据生成) 和 SCRIPT 2 (DFM训练)。")
        return
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # --- 2. 数据准备和编码器 ---
    encoder = LabelEncoder()
    # 使用训练集来拟合编码器
    encoder.fit(df_train_clean['Modulation_Label'].unique())
    print(f"调制类别: {encoder.classes_}, 数量: {NUM_MODULATION_CLASSES}")

    # --- 3. 加载条件扩散模型 (DFM) ---
    print("\n" + "=" * 50)
    print("      加载【扩散模型 DFM】")
    print("=" * 50)

    dfm_model = ConditionalUNet1D(base_channels=32,
                                  num_modulation_classes=NUM_EMBEDDING_CLASSES).to(device)
    try:
        dfm_model.load_state_dict(torch.load(CONDITIONAL_DDPM_MODEL_PATH, map_location=device))
        print(f"✅ 已加载条件DFM模型权重 (CFG-Enabled, {NUM_EMBEDDING_CLASSES} 类): {CONDITIONAL_DDPM_MODEL_PATH}")
    except FileNotFoundError:
        print(f"⚠️ 警告：未找到条件DFM模型权重 {CONDITIONAL_DDPM_MODEL_PATH}")
        dfm_model = None
    except Exception as e:
        print(f"❌ 加载条件DFM模型失败: {e}")
        dfm_model = None

    # --- 4. [基准测试] 评估基础 AMC 模型 ---
    print("\n" + "=" * 50)
    print("      [基准测试] 评估【基础模型】")
    print("=" * 50)

    amc_base_model = AMCClassifier(num_classes=NUM_MODULATION_CLASSES).to(device)
    base_report = None
    base_report_malicious = None  # [!! 新增 !!]
    try:
        amc_base_model.load_state_dict(torch.load(AMC_BASE_PATH, map_location=device))
        print(f"✅ [加载成功] 基础模型权重 {AMC_BASE_PATH}")

        # [核心修改] 在 *合法测试集* (clean_test.csv) 上测试基准
        # 调用新的轻量级测试函数
        base_report = test_amc_accuracy(
            amc_base_model, df_test_clean, encoder, device, title="[基准] 基础模型在 *合法数据* 上的精度"
        )

        # [!! 新增 !!] 在 *恶意测试集* (pgd_malicious_data.csv) 上测试基准
        base_report_malicious = test_amc_accuracy(
            amc_base_model, df_test_pgd, encoder, device, title="[基准] 基础模型在 *恶意数据* 上的精度"
        )

    except FileNotFoundError:
        print(f"❌ [加载失败] 未找到基础模型权重 {AMC_BASE_PATH}，跳过基准测试。")
        amc_base_model = None
    except Exception as e:
        print(f"❌ [加载失败] 加载基础模型出错: {e}")
        amc_base_model = None

    # --- 5. [增强训练] 训练增强 AMC 模型 ---
    print("\n" + "=" * 50)
    print("      训练【增强模型】 (合法训练集 + 合成数据)")
    print("=" * 50)

    # 合并 df_train_clean 和 df_synthetic
    df_amc_train_enhanced = pd.concat([df_train_clean, df_synthetic]).reset_index(drop=True)
    print(
        f"增强训练数据: {len(df_train_clean)} 合法训练 + {len(df_synthetic)} 条件合成 = {len(df_amc_train_enhanced)} 总样本")

    amc_enhanced_model = AMCClassifier(num_classes=NUM_MODULATION_CLASSES).to(device)
    # [核心] 传入 df_amc_train_enhanced (训练) 和 df_val_clean (验证)
    amc_enhanced_model, enhanced_training_log = train_amc_classifier(
        df_amc_train_enhanced, df_val_clean, encoder, amc_enhanced_model,
        epochs=100, batch_size=128, lr=1e-3, device=device,
        save_path=AMC_ENHANCED_PATH
    )

    # --- 6. [对比测试] 评估增强 AMC 模型 ---
    print("\n" + "=" * 50)
    print("      [对比测试] 评估【增强模型】")
    print("=" * 50)

    # [核心修改] 同样在 *合法测试集* (clean_test.csv) 上测试
    # 再次调用新的轻量级测试函数
    enhanced_report = test_amc_accuracy(
        amc_enhanced_model, df_test_clean, encoder, device, title="[对比] 增强模型在 *合法数据* 上的精度"
    )

    # [!! 新增 !!] 在 *恶意测试集* (pgd_malicious_data.csv) 上测试增强模型
    enhanced_report_malicious = test_amc_accuracy(
        amc_enhanced_model, df_test_pgd, encoder, device, title="[对比] 增强模型在 *恶意数据* 上的精度"
    )

    # --- 7. [异常检测] 评估 DFM 性能 ---
    print("\n" + "=" * 80)
    print("                      【DFM 异常检测性能报告】")
    print("=" * 80)

    # [核心] 我们合并 *合法测试集* 和 *恶意测试集* 来创建 DFM 的评估集
    print("  > 正在合并 合法测试集 和 恶意测试集 用于 DFM 评估...")
    df_detection_test_set = pd.concat([df_test_clean, df_test_pgd]).reset_index(drop=True)
    print(f"  > DFM 评估集大小: {len(df_detection_test_set)} 样本")

    # 运行 G-EAMD (仅为获取分数，AMC 模型用哪个都行，这里用增强版)
    # [修改] 确保 amc_enhanced_model 存在 (如果训练失败则回退到 base)
    model_for_geamd = amc_enhanced_model if amc_enhanced_model is not None else amc_base_model

    if model_for_geamd is not None and dfm_model is not None:  # [修改] 确保 DFM 也加载成功
        dfm_results_df, legit_scores, anomaly_scores = evaluate_geamd(
            df_detection_test_set, dfm_model, model_for_geamd, encoder, device
        )
    else:
        print("  > 警告: 缺少 DFM 或 AMC 模型，跳过 G-EAMD 评估。")
        dfm_results_df, legit_scores, anomaly_scores = None, {}, {}

    if dfm_results_df is not None:
        evaluate_anomaly_detection_rate(dfm_results_df, title="DFM 非法信号检测率")
    else:
        print("DFM 评估失败，跳过检测准确率报告。")

    # --- 8. [!! 修改 !!] 最终 AMC 结果对比 ---
    print("\n" + "=" * 70)
    print("                          【AMC 训练结果对比报告】")
    print("=" * 70)

    if base_report and enhanced_report:
        # 打印平均精度对比
        print(">> 合法信号分类性能 (性能增强):")
        print(f"  基础模型 (合法数据) 平均精度:     {base_report['Average_Accuracy']:.2f}%")
        print(f"  增强模型 (合法数据) 平均精度: {enhanced_report['Average_Accuracy']:.2f}%")

        # [!! 新增 !!] 恶意数据上的平均精度
        if base_report_malicious and enhanced_report_malicious:
            print("\n>> 恶意信号分类性能:")
            print(f"  基础模型 (恶意数据) 平均精度:     {base_report_malicious['Average_Accuracy']:.2f}%")
            print(f"  增强模型 (恶意数据) 平均精度: {enhanced_report_malicious['Average_Accuracy']:.2f}%")

        print("-" * 70)

        # 打印 SNR 精度对比
        print(">> 按 SNR 分类的详细精度 (%) [合法数据]")
        print(f"{'SNR (dB)':<10} | {'基础模型':<15} | {'增强模型 (条件DFM)':<15} | {'提升/下降':<15}")
        print("-" * 70)
        snr_keys = sorted(base_report['SNR_Accuracy'].keys())
        for snr in snr_keys:
            base_acc = base_report['SNR_Accuracy'].get(snr, 0.0)
            enhanced_acc = enhanced_report['SNR_Accuracy'].get(snr, 0.0)
            difference = enhanced_acc - base_acc
            print(f"{snr:<10} | {base_acc:<15.2f} | {enhanced_acc:<15.2f} | {difference:<+15.2f}")

        # [!! 新增 !!] 恶意数据上的详细精度
        if base_report_malicious and enhanced_report_malicious:
            print("\n>> 按 SNR 分类的详细精度 (%) [恶意数据]")
            print(f"{'SNR (dB)':<10} | {'基础模型':<15} | {'增强模型 (条件DFM)':<15}")
            print("-" * 70)
            snr_keys_malicious = sorted(list(set(base_report_malicious['SNR_Accuracy'].keys()) | set(
                enhanced_report_malicious['SNR_Accuracy'].keys())))
            for snr in snr_keys_malicious:
                base_acc = base_report_malicious['SNR_Accuracy'].get(snr, 0.0)
                enhanced_acc = enhanced_report_malicious['SNR_Accuracy'].get(snr, 0.0)
                print(f"{snr:<10} | {base_acc:<15.2f} | {enhanced_acc:<15.2f}")
    else:
        print("未能加载基准模型，无法进行性能对比。")

    print("=" * 70)

    # --- 8.5 [!! 新增 !!] DFM 重构误差对比 ---
    print("\n" + "=" * 70)
    print("                       【DFM 异常分数 (重构误差) 对比】")
    print("=" * 70)
    print(">> 该分数用于异常检测。分数越高，代表与合法数据的分布差异越大。")
    print(f"{'SNR (dB)':<10} | {'合法信号 (Clean)':<20} | {'恶意信号 (PGD)':<20}")
    print("-" * 70)

    if legit_scores and anomaly_scores:
        all_dfm_snrs = sorted(list(set(legit_scores.keys()) | set(anomaly_scores.keys())))
        for snr in all_dfm_snrs:
            legit_s = legit_scores.get(snr, 0.0)
            anomaly_s = anomaly_scores.get(snr, 0.0)
            print(f"{snr:<10} | {legit_s:<20.6f} | {anomaly_s:<20.6f}")
    else:
        print("DFM 评估未运行，无分数可报告。")
    print("=" * 70)

    # --- 9. [!! 修改 !!] 保存详细数据 (包括恶意数据上的AMC精度) ---
    import os
    os.makedirs('./figures', exist_ok=True)

    # [!! 修改 !!] 检查所有报告是否存在
    if base_report and enhanced_report and dfm_results_df is not None:
        # 从 DFM 评估中获取合法的分数
        dfm_legit_scores = dfm_results_df[dfm_results_df['True_Anomaly'] == False][['SNR_dB', 'Anomaly_Score']]
        dfm_legit_scores = dfm_legit_scores.groupby('SNR_dB')['Anomaly_Score'].mean().to_dict()

        # 从 DFM 评估中获取非法的分数
        dfm_anomaly_scores = dfm_results_df[dfm_results_df['True_Anomaly'] == True][['SNR_dB', 'Anomaly_Score']]
        dfm_anomaly_scores = dfm_anomaly_scores.groupby('SNR_dB')['Anomaly_Score'].mean().to_dict()

        # [!! 修改 !!] 合并所有SNR键
        all_snrs_amc_legit = set(base_report['SNR_Accuracy'].keys()) | set(enhanced_report['SNR_Accuracy'].keys())
        all_snrs_amc_malicious = set()
        if base_report_malicious:
            all_snrs_amc_malicious.update(base_report_malicious['SNR_Accuracy'].keys())
        if enhanced_report_malicious:
            all_snrs_amc_malicious.update(enhanced_report_malicious['SNR_Accuracy'].keys())

        all_snrs_dfm = set(legit_scores.keys()) | set(anomaly_scores.keys())

        all_snrs = sorted(list(all_snrs_amc_legit | all_snrs_amc_malicious | all_snrs_dfm))

        detailed_snr_data = []
        for snr in all_snrs:
            detailed_snr_data.append({
                'SNR_dB': snr,
                # 合法 AMC 精度
                'Base_Legit_Accuracy': base_report['SNR_Accuracy'].get(snr, 0.0),
                'Enhanced_Legit_Accuracy': enhanced_report['SNR_Accuracy'].get(snr, 0.0),
                # [!! 新增 !!] 恶意 AMC 精度
                'Base_Malicious_Accuracy': base_report_malicious['SNR_Accuracy'].get(snr,
                                                                                     0.0) if base_report_malicious else 0.0,
                'Enhanced_Malicious_Accuracy': enhanced_report_malicious['SNR_Accuracy'].get(snr,
                                                                                             0.0) if enhanced_report_malicious else 0.0,
                # DFM 重构误差 (异常分数)
                'DFM_Legit_Score': dfm_legit_scores.get(snr, 0.0),
                'DFM_Anomaly_Score': dfm_anomaly_scores.get(snr, 0.0)
            })

        detailed_df = pd.DataFrame(detailed_snr_data)
        detailed_df.to_csv('./figures/detailed_snr_performance.csv', index=False)
        print(f"\n详细信噪比性能数据已保存至: ./figures/detailed_snr_performance.csv")
    else:
        print("\n[!!] 警告: 缺少部分评估报告 (DFM, Base, 或 Enhanced)，未保存详细的 snr 性能 csv。")


if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs('./figures', exist_ok=True)

    # [修改] 修正函数调用
    run_amc_eval_stage()