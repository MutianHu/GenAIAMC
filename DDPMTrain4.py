# ====================================================================
# SCRIPT 2: 条件生成扩散模型 (支持 Classifier-Free Guidance)
# [修改版：支持选择加载已有模型 + 验证集监控与早停]
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
import os
from datetime import datetime
import ast

# --- 全局配置 ---
SAMPLES_PER_SIGNAL = 256
CHANNELS = 2
T = 500
DDPM_MODEL_PATH = 'conditional_dfm_unet_model.pth'
SYNTHETIC_DATA_PATH = 'conditional_synthetic_data.csv'
TRAINING_LOG_PATH = 'conditional_ddpm_training_log.csv'

# [新增] 数据集路径
CLEAN_TRAIN_PATH = 'clean_train.csv'
CLEAN_VAL_PATH = 'clean_val.csv'  # [新增] 验证集路径

# [恢复] 回到简单的 Mod-Only 标签
MODULATION_TYPES = ['GFSK', 'QPSK', '16QAM']
NUM_MODULATION_CLASSES = len(MODULATION_TYPES)

# --- [恢复] CFG (Classifier-Free Guidance) 配置 ---
NULL_LABEL_INDEX = NUM_MODULATION_CLASSES  # "空标签"索引 (例如: 3)
NUM_EMBEDDING_CLASSES = NUM_MODULATION_CLASSES + 1  # 总嵌入类别数 (例如: 3个真实类 + 1个空类 = 4)
UNCONDITIONAL_PROB = 0.1  # 10% 的概率丢弃标签，使用 "空标签"


# --- 结束恢复 ---

# --- [删除] 数据划分配置不再需要 ---


# --- DDPM 参数定义 ---
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """线性beta调度"""
    return torch.linspace(start, end, timesteps)


# 定义DDPM参数
betas = linear_beta_schedule(T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


# Extract函数定义
def extract(arr, timesteps, broadcast_shape):
    """
    从数组中根据时间步提取值
    arr: 要提取的数组
    timesteps: 时间步张量
    broadcast_shape: 广播到的形状
    """
    device = arr.device if torch.is_tensor(arr) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.is_tensor(arr):
        arr = torch.from_numpy(arr).to(device)

    # 确保timesteps在正确的设备上
    timesteps = timesteps.to(device)

    # 从arr中根据timesteps索引取值
    res = arr[timesteps].float()

    # 调整形状以匹配broadcast_shape
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)

    return res.expand(broadcast_shape)


# 将参数移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
alphas_cumprod_prev = alphas_cumprod_prev.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)


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


# --- 条件UNet1D模型 ---
class ConditionalUNet1D(nn.Module):
    """
    条件生成扩散模型
    [恢复] 回到简单的 4 类嵌入
    """

    def __init__(self, in_channels=CHANNELS, out_channels=CHANNELS, base_channels=32, num_conv_blocks=2,
                 num_embedding_classes=NUM_EMBEDDING_CLASSES):  # <-- [恢复]
        super().__init__()
        self.num_conv_blocks = num_conv_blocks

        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_channels * 4), nn.GELU(), nn.Linear(base_channels * 4, base_channels * 4)
        )

        # --- [恢复] 调制标签嵌入 (使用 NUM_EMBEDDING_CLASSES) ---
        self.modulation_embed = nn.Sequential(
            nn.Embedding(num_embedding_classes, base_channels * 4),  # <-- 恢复
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )
        # --- 结束恢复 ---

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

    def forward(self, x, t, modulation_labels):  # <-- [恢复]
        # 时间嵌入
        t_emb = self.time_embed(t.unsqueeze(1))

        # [恢复] 调制标签嵌入
        mod_emb = self.modulation_embed(modulation_labels)

        # 合并条件
        condition = torch.cat([t_emb, mod_emb], dim=1)
        condition = self.condition_merge(condition)

        # ... (后续 forward 逻辑不变) ...
        batch_size, _, seq_len = x.shape
        condition_expanded = condition.unsqueeze(-1).expand(-1, -1, seq_len)

        x = self.conv_in(x)
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


# --- [恢复] 简单的条件训练函数 (支持 CFG) ---
def get_conditional_ddpm_loss(model, x_0, modulation_labels):  # <-- [恢复]
    """条件DDPM损失函数 (支持 CFG)"""
    t = torch.randint(0, T, (x_0.shape[0],), device=x_0.device).long()
    noise = torch.randn_like(x_0)
    sqrt_alpha_bar_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha_bar_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    time_input = t.float() / T * 999.0

    # --- CFG 修改：随机丢弃标签 ---
    mask = (torch.rand(x_0.shape[0], device=x_0.device) > UNCONDITIONAL_PROB)
    labels_to_use = modulation_labels.clone()  # <-- [恢复]
    labels_to_use[~mask] = NULL_LABEL_INDEX  # <-- [恢复]
    # --- 结束修改 ---

    # 传入经过 CFG 处理的调制标签
    noise_pred = model(x_t, time_input, labels_to_use)
    return F.mse_loss(noise, noise_pred)


# --- [恢复] 简单的数据集类 (仅 Modulation_Label) ---
class ConditionalSignalDataset(Dataset):
    def __init__(self, df, label_encoder, phase='dfm_train'):
        # 安全地处理I/Q数据
        I_data = []
        Q_data = []
        modulation_labels = []  # <-- [恢复]

        # [修改] 使用 .iterrows() 遍历 df
        for idx, row in df.iterrows():
            # ... (I/Q 数据加载不变) ...
            i_val = row['I']
            if isinstance(i_val, str):
                try:
                    i_array = ast.literal_eval(i_val)
                except:
                    i_array = np.zeros(SAMPLES_PER_SIGNAL)
            else:
                i_array = i_val
            q_val = row['Q']
            if isinstance(q_val, str):
                try:
                    q_array = ast.literal_eval(q_val)
                except:
                    q_array = np.zeros(SAMPLES_PER_SIGNAL)
            else:
                q_array = q_val
            i_array = np.array(i_array, dtype=np.float32)
            q_array = np.array(q_array, dtype=np.float32)
            if len(i_array) != SAMPLES_PER_SIGNAL:
                i_array = np.resize(i_array, SAMPLES_PER_SIGNAL)
            if len(q_array) != SAMPLES_PER_SIGNAL:
                q_array = np.resize(q_array, SAMPLES_PER_SIGNAL)
            I_data.append(i_array)
            Q_data.append(q_array)

            # [恢复] 使用 'Modulation_Label'
            modulation_labels.append(row['Modulation_Label'])

            # 堆叠数据
        I_data = np.stack(I_data)
        Q_data = np.stack(Q_data)
        self.data = np.stack([I_data, Q_data], axis=1)

        # [恢复] 编码调制标签
        self.modulation_labels_str = modulation_labels
        self.modulation_labels_int = label_encoder.transform(modulation_labels)

        print(f"条件数据集({phase})加载完成: {len(self.data)} 个样本, 调制类型: {np.unique(modulation_labels)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.float32),
                torch.tensor(self.modulation_labels_int[idx], dtype=torch.long))  # <-- [恢复]


# --- [修改] 包含验证集的训练函数 ---
def train_conditional_dfm(df_legal_train, df_legal_val, label_encoder, model, epochs=10, batch_size=64, lr=1e-4,
                          device='cuda'):
    print(f"开始训练条件DDPM模型 (支持 CFG)...")

    # 创建训练集和验证集 DataLoader
    dataset_train = ConditionalSignalDataset(df_legal_train, label_encoder, phase='dfm_train')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    # [新增] 验证集
    dataset_val = ConditionalSignalDataset(df_legal_val, label_encoder, phase='dfm_val')
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_log = []
    start_time = datetime.now()

    # [新增] 最佳模型跟踪与早停
    best_val_loss = float('inf')
    patience = 300  # [配置] 早停耐心值
    patience_counter = 0

    print(f"\n训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"数据集: 训练集 {len(dataset_train)}, 验证集 {len(dataset_val)}")
    print(f"调制类别: {label_encoder.classes_}")  # <-- [恢复]

    # --- [新增] 打印 CFG 配置 ---
    print(f"CFG 配置: 启用 (丢弃概率={UNCONDITIONAL_PROB * 100}%, 空标签索引={NULL_LABEL_INDEX})")
    # --- 结束新增 ---

    print("-" * 80)

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        epoch_losses = []

        for data, modulation_labels in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):  # <-- [恢复]
            x_0 = data.to(device)
            mod_labels = modulation_labels.to(device)  # <-- [恢复]

            # 调用修改后的 loss 函数 (CFG 逻辑在函数内部处理)
            loss = get_conditional_ddpm_loss(model, x_0, mod_labels)  # <-- [恢复]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_epoch_loss = np.mean(epoch_losses)

        # --- [新增] 验证阶段 ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data, modulation_labels in dataloader_val:
                x_0 = data.to(device)
                mod_labels = modulation_labels.to(device)
                # 验证时同样计算 loss
                loss = get_conditional_ddpm_loss(model, x_0, mod_labels)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        # --- [新增] 最佳模型保存与早停 ---
        save_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), DDPM_MODEL_PATH)
            save_msg = " [* New Best *]"
        else:
            patience_counter += 1

        # 记录训练日志
        log_entry = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_loss': avg_epoch_loss,
            'val_loss': avg_val_loss,  # [新增]
            'learning_rate': lr,
            'batch_size': batch_size
        }
        training_log.append(log_entry)

        log_df = pd.DataFrame(training_log)
        log_df.to_csv(TRAINING_LOG_PATH, index=False)

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | Train Loss: {avg_epoch_loss:.6f} | Val Loss: {avg_val_loss:.6f}{save_msg}")

        # 早停检查
        if patience_counter >= patience:
            print(f"\n[早停] 验证损失在 {patience} 个 epoch 内未下降。停止训练。")
            break

    # 训练完成
    end_time = datetime.now()
    training_duration = end_time - start_time

    print("\n" + "=" * 80)
    print("条件DDPM (CFG) 模型训练完成!")
    print(f"训练总时长: {training_duration}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最佳模型已保存至 {DDPM_MODEL_PATH}")

    # [关键] 重新加载最佳模型权重，以供后续生成使用
    model.load_state_dict(torch.load(DDPM_MODEL_PATH, map_location=device))

    return model, training_log


# --- [核心修改] 条件样本生成函数 (混合 t_stop) ---
@torch.no_grad()
def generate_conditional_samples(model, label_encoder, num_samples_per_level, t_stop_levels, batch_size=512,
                                 device='cuda',
                                 guidance_scale=7.0):
    """
    为每个调制类型生成 *带噪* 样本
    通过在 t_stop > 0 时停止去噪循环
    [修改] 接收一个 t_stop_levels 列表，并为每个 level 生成 num_samples_per_level 个样本
    """
    print(f"开始条件生成样本 (混合 t_stop, w={guidance_scale})")
    print(f"  > t_stop 级别: {t_stop_levels}")
    print(f"  > 每个级别/每种调制的样本数: {num_samples_per_level}")
    model.eval()

    all_synthetic_data = []
    all_modulation_labels = []
    all_snr_labels = []  # [新增] 用于存储 t_stop

    # [核心修改] 外循环：遍历 t_stop 级别
    for t_stop in t_stop_levels:
        print(f"\n--- 正在生成 t_stop = {t_stop} ---")

        # [恢复] 内循环：遍历调制类型
        for modulation_type in MODULATION_TYPES:
            print(f"  > 生成 {modulation_type} @ t_stop={t_stop}...")
            modulation_int = label_encoder.transform([modulation_type])[0]

            synthetic_results = []
            num_batches = (num_samples_per_level + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(num_batches), desc=f"    > {modulation_type}"):
                current_batch_size = min(batch_size, num_samples_per_level - len(synthetic_results))
                if current_batch_size <= 0:
                    break

                # 从噪声开始
                x = torch.randn(current_batch_size, CHANNELS, SAMPLES_PER_SIGNAL, device=device)

                # [恢复] 创建两种标签：
                modulation_labels_cond = torch.full((current_batch_size,), modulation_int,
                                                    device=device, dtype=torch.long)
                modulation_labels_uncond = torch.full((current_batch_size,), NULL_LABEL_INDEX,
                                                      device=device, dtype=torch.long)

                # [核心修改] 修改循环以提前停止
                # 循环从 T-1 到 t_stop
                for t in reversed(range(t_stop, T)):
                    t_tensor = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
                    time_input = t_tensor.float() / T * 999.0

                    # --- CFG 预测 ---
                    noise_pred_cond = model(x, time_input, modulation_labels_cond)
                    noise_pred_uncond = model(x, time_input, modulation_labels_uncond)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    # --- 结束 ---

                    alpha_t = extract(alphas, t_tensor, x.shape)
                    sqrt_one_minus_alpha_bar_t = extract(sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)

                    mean = 1 / torch.sqrt(alpha_t) * (
                            x - (betas[t_tensor.cpu()].to(device).view(-1, 1,
                                                                       1) / sqrt_one_minus_alpha_bar_t) * noise_pred)

                    if t > t_stop:  # 只要不是最后一步，就添加噪声
                        variance = betas[t_tensor.cpu()].to(device).view(-1, 1, 1)
                        x = mean + torch.sqrt(variance) * torch.randn_like(x)
                    else:  # 这是最后一步 (t == t_stop)
                        x = mean  # 不再添加噪声，返回 x_{t_stop}

                # [修改] 现在的 x 是 x_{t_stop} (带噪信号)
                generated_data = x.cpu().numpy()
                synthetic_results.append(generated_data)

            if synthetic_results:
                class_synthetic_data = np.concatenate(synthetic_results, axis=0)
                all_synthetic_data.append(class_synthetic_data)
                all_modulation_labels.extend([modulation_type] * len(class_synthetic_data))
                all_snr_labels.extend([f't_stop_{t_stop}'] * len(class_synthetic_data))  # [修改] 保存 t_stop 标签

    if not all_synthetic_data:
        return pd.DataFrame()

    # 合并所有类别的数据
    final_synthetic_data = np.concatenate(all_synthetic_data, axis=0)

    synthetic_df = pd.DataFrame({
        'I': list(final_synthetic_data[:, 0, :]),
        'Q': list(final_synthetic_data[:, 1, :]),
        'Modulation_Label': all_modulation_labels,
        'Anomaly_Label': ['Legit'] * len(final_synthetic_data),
        'SNR_dB': all_snr_labels,  # [修改]
        'Fd_max_Hz': [0.0] * len(final_synthetic_data),
        'Data_Source': ['Synthetic'] * len(final_synthetic_data)
    })

    print(f"\n生成完成: 总样本数 {len(final_synthetic_data)}")
    print("按调制类型统计:")
    print(synthetic_df['Modulation_Label'].value_counts())
    print("按 t_stop (SNR 代理) 统计:")
    print(synthetic_df['SNR_dB'].value_counts())

    return synthetic_df


# --- [核心修改] 主函数 (run_conditional_ddpm_stage) ---
def run_conditional_ddpm_stage():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # [修改] 加载预划分的 *训练集* 和 *验证集*
    try:
        print(f"正在加载合法训练集: {CLEAN_TRAIN_PATH}...")
        df_train_clean = pd.read_csv(CLEAN_TRAIN_PATH)
        print(f"正在加载合法验证集: {CLEAN_VAL_PATH}...")
        df_val_clean = pd.read_csv(CLEAN_VAL_PATH)  # [新增]

        print(f"成功加载数据: 训练集 {len(df_train_clean)}, 验证集 {len(df_val_clean)}")

        # 创建标签编码器
        label_encoder = LabelEncoder()
        label_encoder.fit(MODULATION_TYPES)
        print(f"调制类别: {label_encoder.classes_}")

    except FileNotFoundError as e:
        print(f"错误: 未找到数据文件 ({e})")
        print("请确保您已经先运行了 SCRIPT 1 (data_gen_pgd_stage.py) 来生成预划分的数据集。")
        return
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # 初始化模型结构
    # [修改] 我们需要使用与 SCRIPT 3 中 *相同* 的参数初始化 DFM
    conditional_model = ConditionalUNet1D(
        base_channels=32,
        num_embedding_classes=NUM_EMBEDDING_CLASSES  # <-- 确保传入 4
    ).to(device)

    # --- [新增逻辑] 检查模型是否存在并询问是否直接加载 ---
    should_train = True
    if os.path.exists(DDPM_MODEL_PATH):
        print("\n" + "=" * 60)
        print(f"发现已保存的模型权重文件: {DDPM_MODEL_PATH}")
        user_choice = input("是否直接加载该模型并跳过训练? (输入 y 加载, 输入其他键重新训练): ").strip().lower()
        if user_choice == 'y':
            try:
                conditional_model.load_state_dict(torch.load(DDPM_MODEL_PATH, map_location=device))
                print(">>> 模型权重加载成功！跳过训练阶段。")
                should_train = False
            except Exception as e:
                print(f">>> 加载模型失败 ({e})，将开始重新训练。")
                should_train = True
        else:
            print(">>> 用户选择重新训练。")
    else:
        print(f"\n未找到已保存的模型 {DDPM_MODEL_PATH}，开始新训练。")

    # 如果需要训练
    if should_train:
        conditional_model, training_log = train_conditional_dfm(
            df_train_clean, df_val_clean, label_encoder, conditional_model, epochs=300, device=device
            # [修改] 传入 df_val_clean
        )

    # --- [核心修改] 定义混合 t_stop 策略 ---
    T_STOP_LEVELS = [20, 15, 10, 5, 0]  # <-- 您建议的测试级别
    print(f"\n将为 {len(T_STOP_LEVELS)} 个 t_stop 级别生成数据: {T_STOP_LEVELS}")

    # [修改] 计算每个级别要生成的样本数
    num_samples_per_class_total = len(df_train_clean)*2 // len(MODULATION_TYPES)
    num_samples_per_level = num_samples_per_class_total // len(T_STOP_LEVELS)
    print(f"  > 原始训练集每个调制类约有 {num_samples_per_class_total} 个样本")
    print(f"  > 将为每个级别/每个调制类生成 {num_samples_per_level} 个样本")

    # [核心修改] 调用增强版的生成函数
    synthetic_data = generate_conditional_samples(
        conditional_model,
        label_encoder,
        num_samples_per_level,
        T_STOP_LEVELS,  # <--- 传入级别列表
        device=device,
        guidance_scale=7.0  # <-- 保持 7.0
    )

    # 保存合成数据
    synthetic_data.to_csv(SYNTHETIC_DATA_PATH, index=False)
    print(f"条件合成数据已保存至 {SYNTHETIC_DATA_PATH}")

    # 返回模型和日志（如果未训练，日志为None）
    training_log = None if not should_train else locals().get('training_log')
    return conditional_model, training_log


if __name__ == '__main__':
    os.makedirs('./figures', exist_ok=True)
    run_conditional_ddpm_stage()