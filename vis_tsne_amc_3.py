import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import ast
import os
import copy
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# ================= Configuration Area =================
# Path to clean data
CLEAN_TEST_PATH = 'clean_test.csv'

# Model paths
BASE_MODEL_PATH = 'amc_base_model.pth'
ENHANCED_MODEL_PATH = 'amc_enhanced_model.pth'

# PGD Attack Parameters (for on-the-fly generation)
PGD_EPS = 0.1  # Perturbation magnitude
PGD_ALPHA = 0.02  # Step size
PGD_STEPS = 10  # Number of iterations

# Sampling settings
MIN_SNR_DB = 5
SAMPLES_PER_CLASS = 300  # Number of samples per class

# Signal parameters
SAMPLES_PER_SIGNAL = 256
CHANNELS = 2
MODULATION_TYPES = ['GFSK', 'QPSK', '16QAM']
NUM_CLASSES = len(MODULATION_TYPES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- [Nature Color Scheme] ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# Nature Style: [Blue, Green, Red] corresponding to [GFSK, QPSK, 16QAM]
# Using red instead of orange as per your preference from image_5.png
NATURE_COLORS = ['#699ECA', '#4DAF4A', '#D62728']

# --- [Aesthetic Modifications] ---
# 1. Lighter Background: Reduced alpha from 0.2 to 0.1
BG_ALPHA = 0.1

# Marker styles
MARKER_CLEAN = 'o'
MARKER_SIZE_CLEAN = 55
EDGE_COLOR_CLEAN = 'white'
EDGE_WIDTH_CLEAN = 0.9

MARKER_ADV = '*'
MARKER_SIZE_ADV = 100
EDGE_COLOR_ADV = '#333333'
EDGE_WIDTH_ADV = 0.4


# ===========================================

# --- 1. Model Definition (Fixed Forward Method) ---
class AMCClassifier(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.norm2 = nn.BatchNorm1d(out_channels)
            if in_channels != out_channels or stride != 1:
                self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            else:
                self.residual_conv = nn.Identity()

        def forward(self, x):
            h = F.relu(self.norm1(self.conv1(x)))
            h = self.norm2(self.conv2(h))
            return F.relu(h + self.residual_conv(x))

    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(CHANNELS, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            self.ResidualBlock(64, 64, stride=2),
            self.ResidualBlock(64, 64, stride=2),
            self.ResidualBlock(64, 64, stride=2),
            self.ResidualBlock(64, 64, stride=2),
            self.ResidualBlock(64, 64)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )

    # [Fixed] Added forward method back for PGD attack loss calculation
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).squeeze(-1)
        return self.classifier(x)

    # Keep extract_features method for t-SNE feature extraction
    def extract_features(self, x):
        x = self.features(x)
        x = self.avgpool(x).squeeze(-1)
        return x


# --- 2. Dataset Utility ---
class SimpleSignalDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.data = []
        self.labels = []
        # Pre-process data
        for _, row in df.iterrows():
            try:
                # Handle potential string format
                i_val = ast.literal_eval(row['I']) if isinstance(row['I'], str) else row['I']
                q_val = ast.literal_eval(row['Q']) if isinstance(row['Q'], str) else row['Q']

                signal = np.stack([i_val, q_val], axis=0).astype(np.float32)
                # Length alignment
                if signal.shape[1] != SAMPLES_PER_SIGNAL:
                    temp = np.zeros((2, SAMPLES_PER_SIGNAL), dtype=np.float32)
                    min_len = min(SAMPLES_PER_SIGNAL, signal.shape[1])
                    temp[:, :min_len] = signal[:, :min_len]
                    signal = temp
                self.data.append(signal)
                self.labels.append(row['Modulation_Label'])
            except Exception as e:
                continue

        self.data = np.array(self.data)
        if len(self.labels) > 0:
            self.labels_int = label_encoder.transform(self.labels)
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.labels_int = torch.tensor(self.labels_int, dtype=torch.long)
        else:
            self.data = torch.empty(0)
            self.labels_int = torch.empty(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels_int[idx]


# --- 3. On-the-fly PGD Attack Function ---
def generate_adv_on_the_fly(model, df_clean, label_encoder):
    """
    Receives a DataFrame of clean data and generates a corresponding DataFrame of adversarial samples using the current model.
    """
    model.eval()
    adv_data_list = []

    # Convert DataFrame to Dataset for batch processing
    dataset = SimpleSignalDataset(df_clean, label_encoder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(f"  > Generating adversarial samples (Batch PGD, eps={PGD_EPS})...")

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        # --- PGD Attack Start ---
        x = data.clone().detach()
        # Random initialization
        delta = torch.zeros_like(x).uniform_(-PGD_EPS, PGD_EPS).to(DEVICE)
        delta.requires_grad = True

        for _ in range(PGD_STEPS):
            adv_x = x + delta
            # Call model(adv_x), now with forward method, no error will occur
            output = model(adv_x)
            loss = F.cross_entropy(output, target)

            loss.backward()
            grad_sign = delta.grad.detach().sign()

            delta.data = delta.data + PGD_ALPHA * grad_sign
            delta.data = torch.clamp(delta.data, -PGD_EPS, PGD_EPS)
            delta.grad.zero_()

        adv_x = x + delta.detach()
        # --- PGD Attack End ---

        # Store generated adversarial samples back to list
        adv_x_np = adv_x.cpu().numpy()
        labels_np = target.cpu().numpy()

        for i in range(len(adv_x_np)):
            # For rigor, we construct new records directly
            # We need to get original SNR information, though not used in plotting stage (as already filtered), to keep structure consistent
            orig_idx = batch_idx * 32 + i
            orig_row = df_clean.iloc[orig_idx]

            adv_data_list.append({
                'I': adv_x_np[i][0],
                'Q': adv_x_np[i][1],
                'Modulation_Label': label_encoder.inverse_transform([labels_np[i]])[0],
                'SNR_dB': orig_row['SNR_dB']
            })

    return pd.DataFrame(adv_data_list)


# --- 4. Plotting Helper Function ---
def plot_decision_boundary_knn(ax, X_2d, y, resolution=0.05, padding=0.5):
    clf = KNeighborsClassifier(n_neighbors=50, weights='uniform')
    clf.fit(X_2d, y)

    x_min, x_max = X_2d[:, 0].min() - padding, X_2d[:, 0].max() + padding
    y_min, y_max = X_2d[:, 1].min() - padding, X_2d[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_background = ListedColormap(NATURE_COLORS)
    ax.contourf(xx, yy, Z, cmap=cmap_background, alpha=BG_ALPHA)

    # 2. Thicker Boundary Lines
    ax.contour(xx, yy, Z, colors='k', linewidths=0.8, alpha=0.3)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


# --- [新增] 智能取整函数 (最大余额法) ---
def get_smart_rounded_cm(y_true, y_pred):
    """
    计算混淆矩阵，并使用最大余额法(Largest Remainder Method)
    确保每一行的百分比之和严格等于 100。
    """
    # 1. 计算原始归一化矩阵 (0.0 - 1.0)
    cm_float = confusion_matrix(y_true, y_pred, normalize='true')

    # 2. 转换为 0 - 100 的数值
    cm_percent = cm_float * 100

    # 3. 初始化结果矩阵
    n_rows, n_cols = cm_percent.shape
    cm_rounded = np.zeros_like(cm_percent, dtype=int)

    # 4. 逐行应用最大余额法
    for i in range(n_rows):
        row = cm_percent[i]

        # 向下取整
        floored = np.floor(row).astype(int)

        # 计算小数部分 (余数)
        remainders = row - floored

        # 计算当前总和与 100 的差值
        current_sum = np.sum(floored)
        diff = 100 - current_sum

        # 将余数从大到小排序，获取索引
        # argsort 默认是升序，[::-1] 变为降序
        sorted_indices = np.argsort(remainders)[::-1]

        # 将差值 (diff) 分配给余数最大的那几项
        for j in range(int(diff)):
            idx = sorted_indices[j]
            floored[idx] += 1

        cm_rounded[i] = floored

    return cm_rounded


# --- [修改] 混淆矩阵绘图函数 (应用智能取整 + 大字体) ---
def plot_confusion_matrices(title_prefix, y_true_clean, y_pred_clean, y_true_adv, y_pred_adv, class_names, save_dir):
    """
    绘制并保存 Side-by-Side 的混淆矩阵 (Clean vs Adversarial)
    使用 Nature 自定义渐变色，增大字体，并显示严格加和为100的百分比
    """

    # 定义自定义渐变色
    cmap_clean = LinearSegmentedColormap.from_list("NatureBlue", ["#F0F8FF", NATURE_COLORS[0]])
    cmap_adv = LinearSegmentedColormap.from_list("NatureRed", ["#FFF5F5", NATURE_COLORS[2]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # --- 1. Clean Samples CM ---
    # 使用智能取整获取矩阵 (内容是 0-100 的整数)
    cm_clean_int = get_smart_rounded_cm(y_true_clean, y_pred_clean)

    # values_format='d' 显示整数
    disp_clean = ConfusionMatrixDisplay(confusion_matrix=cm_clean_int, display_labels=class_names)
    disp_clean.plot(ax=axes[0], cmap=cmap_clean, colorbar=False, values_format='d')

    axes[0].set_title(f"Clean Samples", fontsize=23, weight='bold', pad=15)
    axes[0].tick_params(axis='both', which='major', labelsize=23)
    axes[0].set_xlabel('Predicted label', fontsize=23)
    axes[0].set_ylabel('True label', fontsize=23)

    # [核心修改] 遍历文本：1. 添加'%'符号 2. 设置字体为23
    for text in disp_clean.text_.ravel():
        original_val = text.get_text()
        text.set_text(f"{original_val}%")
        text.set_fontsize(23)

    # --- 2. Adversarial Samples CM ---
    # 使用智能取整获取矩阵
    cm_adv_int = get_smart_rounded_cm(y_true_adv, y_pred_adv)

    disp_adv = ConfusionMatrixDisplay(confusion_matrix=cm_adv_int, display_labels=class_names)
    disp_adv.plot(ax=axes[1], cmap=cmap_adv, colorbar=False, values_format='d')

    axes[1].set_title(f"Adversarial Samples", fontsize=23, weight='bold', pad=15)
    axes[1].tick_params(axis='both', which='major', labelsize=23)
    axes[1].set_xlabel('Predicted label', fontsize=23)
    axes[1].set_ylabel('True label', fontsize=23)

    # [核心修改] 遍历文本：1. 添加'%'符号 2. 设置字体为23
    for text in disp_adv.text_.ravel():
        original_val = text.get_text()
        text.set_text(f"{original_val}%")
        text.set_fontsize(23)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"confusion_matrix_{title_prefix.split()[0].lower()}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"  > Confusion Matrix saved: {save_path}")
    plt.close()


# --- 5. Main Visualization Logic ---
def visualize_pipeline(model_path, title_prefix, df_all_clean, encoder):
    print(f"\nProcessing {model_path} ...")

    # 1. Load model
    model = AMCClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Warning: Model {model_path} not found")
        return

    # 2. Filter data (only high SNR)
    df_clean_filtered = df_all_clean[df_all_clean['SNR_dB'] >= MIN_SNR_DB].copy()

    # 3. Sample (N per class)
    sampled_dfs = []
    for mod in MODULATION_TYPES:
        temp = df_clean_filtered[df_clean_filtered['Modulation_Label'] == mod]
        if len(temp) > SAMPLES_PER_CLASS:
            temp = temp.sample(n=SAMPLES_PER_CLASS, random_state=42)
        sampled_dfs.append(temp)

    if not sampled_dfs:
        print("Not enough data.")
        return

    df_clean_subset = pd.concat(sampled_dfs).reset_index(drop=True)
    print(f"  > Clean samples sampled: {len(df_clean_subset)} (approx. {SAMPLES_PER_CLASS} per class)")

    # 4. Generate corresponding adversarial samples on-the-fly
    df_adv_subset = generate_adv_on_the_fly(model, df_clean_subset, encoder)
    print(f"  > Corresponding adversarial samples generated: {len(df_adv_subset)}")

    # --- Calculate Accuracy ---
    def get_predictions(df):
        dataset = SimpleSignalDataset(df, encoder)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(DEVICE)
                output = model(x)  # Forward pass
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
        return np.array(all_labels), np.array(all_preds)

    print("  > Calculating Accuracy...")
    y_true_clean, y_pred_clean = get_predictions(df_clean_subset)
    y_true_adv, y_pred_adv = get_predictions(df_adv_subset)

    acc_clean = accuracy_score(y_true_clean, y_pred_clean)
    acc_adv = accuracy_score(y_true_adv, y_pred_adv)

    print("-" * 40)
    print(f"  [METRICS] {title_prefix}")
    print(f"  > Clean Data Accuracy:       {acc_clean * 100:.2f}%")
    print(f"  > Adversarial Data Accuracy: {acc_adv * 100:.2f}%")
    print("-" * 40)

    # --- Plot Confusion Matrix (Strict Percentage) ---
    save_dir = r"D:\work_hu\GFFD\paper_visualizations"
    os.makedirs(save_dir, exist_ok=True)
    plot_confusion_matrices(title_prefix, y_true_clean, y_pred_clean, y_true_adv, y_pred_adv, encoder.classes_,
                            save_dir)

    # 5. Extract features
    def get_features(df):
        dataset = SimpleSignalDataset(df, encoder)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        feats, labs = [], []
        model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                f = model.extract_features(x.to(DEVICE))
                feats.append(f.cpu().numpy())
                labs.append(y.numpy())
        return np.concatenate(feats), np.concatenate(labs)

    print("  > Extracting features...")
    X_clean, y_clean = get_features(df_clean_subset)
    X_adv, y_adv = get_features(df_adv_subset)

    # 6. t-SNE
    print("  > Running t-SNE...")
    X_combined = np.concatenate([X_clean, X_adv], axis=0)
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, init='pca', learning_rate='auto')
    X_embedded_all = tsne.fit_transform(X_combined)

    n_clean = len(X_clean)
    X_2d_clean = X_embedded_all[:n_clean]
    X_2d_adv = X_embedded_all[n_clean:]

    # 7. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    class_names = encoder.classes_

    # Left: Clean
    ax0 = axes[0]
    plot_decision_boundary_knn(ax0, X_2d_clean, y_clean)
    for i, c in enumerate(class_names):
        idx = y_clean == i
        ax0.scatter(
            X_2d_clean[idx, 0], X_2d_clean[idx, 1],
            c=NATURE_COLORS[i], label=c,
            marker=MARKER_CLEAN, s=MARKER_SIZE_CLEAN,
            alpha=0.9, edgecolors=EDGE_COLOR_CLEAN, linewidth=EDGE_WIDTH_CLEAN
        )
    ax0.set_title(f"{title_prefix}: Clean Samples", fontsize=18, weight='bold', pad=15)
    ax0.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9, fontsize=22)
    ax0.set_xticks([]);
    ax0.set_yticks([])

    # Right: Adv
    ax1 = axes[1]
    plot_decision_boundary_knn(ax1, X_2d_clean, y_clean)
    for i, c in enumerate(class_names):
        idx = y_adv == i
        ax1.scatter(
            X_2d_adv[idx, 0], X_2d_adv[idx, 1],
            c=NATURE_COLORS[i], label=f"Adversarial {c}",
            marker=MARKER_ADV, s=MARKER_SIZE_ADV,
            alpha=0.9, edgecolors=EDGE_COLOR_ADV, linewidth=EDGE_WIDTH_ADV
        )
    ax1.set_title(f"{title_prefix}: Adversarial Samples", fontsize=18, weight='bold', pad=15)
    ax1.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9, fontsize=22)
    ax1.set_xticks([]);
    ax1.set_yticks([])

    plt.tight_layout()

    # Save
    file_basename = f"tsne_onthefly_{title_prefix.split()[0].lower()}"

    # PNG
    plt.savefig(os.path.join(save_dir, file_basename + ".png"), bbox_inches='tight')
    # EPS
    plt.savefig(os.path.join(save_dir, file_basename + ".eps"), bbox_inches='tight', format='eps')

    print(f"  > Results saved to {save_dir}")
    plt.close()


# --- Main Program ---
if __name__ == "__main__":
    # Initialize encoder
    encoder = LabelEncoder()
    encoder.fit(MODULATION_TYPES)

    # 1. Load all clean data
    if not os.path.exists(CLEAN_TEST_PATH):
        print(f"Error: {CLEAN_TEST_PATH} not found")
        exit()

    print("Loading raw test data...")
    df_all_clean = pd.read_csv(CLEAN_TEST_PATH)
    df_all_clean = df_all_clean[df_all_clean['Modulation_Label'].isin(MODULATION_TYPES)]

    # 2. Visualize Base Model
    visualize_pipeline(BASE_MODEL_PATH, "Base_Model", df_all_clean, encoder)

    # 3. Visualize Enhanced Model
    visualize_pipeline(ENHANCED_MODEL_PATH, "Enhanced_Model", df_all_clean, encoder)

    print("\nAll processes finished.")