# G-EAMD: Generative-Enhanced Adversarial Modulation Classification and Detection

This repository contains the core implementation used in the paper **“Automatic Signal Detection and Classification for Physical Layer Security in Low-Altitude Wireless Networks.”**

G-EAMD uses diffusion models in two complementary ways:

1. **Diverse adversarial hardening:** a source-AMC-guided diffusion attack generator learns transferable adversarial perturbations, which are used to harden an architecture-distinct surrogate AMC model.
2. **Modulation–manifold consistency detection (MMCD):** a class-conditional diffusion model learns clean modulation manifolds and verifies whether the deployed AMC prediction is supported by the received waveform.

The current release focuses on the **main training pipeline, default hyperparameters, and reproducible execution order**. Additional evaluation, plotting, environment-locking, and code-cleanup files will continue to be organized and added.

---

## 1. Repository Status

The main implementation and its key experimental settings are already included. In particular, the current release provides:

- A2G clean-signal dataset generation;
- source/target AMC training;
- architecture-distinct surrogate AMC training;
- clean class-conditional diffusion training for MMCD;
- source-guided adversarial diffusion training;
- diverse offline adversarial hardening, including PGD and diffusion baselines.

Some auxiliary evaluation/plotting utilities and repository packaging files will be further cleaned up and added. The default hyperparameters used by the current scripts are exposed through command-line arguments and, where applicable, are also stored in model checkpoints or run logs.

> **Filename note.** The script names below use the intended repository names. If a local copy contains timestamp or revision suffixes such as `(6)` or `(20260914-071927)`, use the corresponding local filename.

---

## 2. Main Pipeline

The recommended execution order is:

```text
A2G dataset generation
        |
        +--> Train source AMC (TargetAMC in code)
        |
        +--> Train surrogate AMC
        |
        +--> Train clean class-conditional diffusion detector
        |
        +--> Train source-guided adversarial diffusion generator
        |
        +--> Construct adversarial banks and run offline hardening
```

### Terminology

The code retains several historical class names:

- `TargetAMC` in the code corresponds to the **fixed source AMC** used to guide adversarial generation in the manuscript.
- `SurrogateAMC` is the **architecture-distinct defense AMC** that is hardened and used for deployment.
- `C-Target`, `C-Surrogate`, `C-Diff`, and `C-Eval` denote independently generated clean datasets for source-model training, surrogate-model training, diffusion-attack learning, and held-out evaluation, respectively.

---

## 3. Core Scripts

### `pipeline_common_0909.py`

Shared definitions for the receiver-side A2G pipeline, including:

- signal and channel configuration;
- modulation generation;
- A2G propagation and receiver preprocessing;
- dataset construction;
- AMC architectures;
- normalization, attacks, checkpoint utilities, and common training functions.

Current signal configuration:

```text
Frame length              512 complex baseband samples
Samples per symbol        8
Symbols per frame         64
Sampling rate             200 kHz
Carrier frequency         2.4 GHz
Modulations               BPSK, QPSK, 8PSK, 16QAM, 64QAM, GFSK
SNR grid                  -10, -5, 0, 5, 10, 15, 20, 25, 30 dB
UAV speeds                0, 5, 10, 15 m/s
Rician K-factor           3, 10 dB
Residual frequency offset bounded within ±100 Hz
```

The saved I/Q frames represent the receiver-side digital domain after A2G propagation, AWGN, synchronization impairment modeling, and receiver gain normalization. Adversarial perturbations are added to these receiver-side digital I/Q samples.

### `pipeline_common_0909_fixed.py`

Compatibility/path-binding version of the shared pipeline used by the later adversarial-diffusion and offline-hardening stages. Keep this file in the repository root together with the other scripts when reproducing the current release.

---

### `01_generate_clean_datasets_0909.py`

Generates four statistically independent clean A2G datasets:

```text
C-Diff       diffusion-attack learning
C-Target     source AMC and clean diffusion detector
C-Surrogate  surrogate AMC
C-Eval       held-out evaluation
```

For each modulation/SNR/UAV-speed/Rician-K combination, the default numbers of samples are:

```text
train: 240
val:    80
test:   80
```

This gives, for each dataset:

```text
Training frames:    103,680
Validation frames:   34,560
Test frames:         34,560
Total:              172,800
```

Run:

```bash
python 01_generate_clean_datasets_0909.py
```

To regenerate existing artifacts:

```bash
python 01_generate_clean_datasets_0909.py --overwrite
```

Generated datasets are stored under `artifacts/data/`.

---

### `03_train_target_amc_0909.py`

Trains the fixed source AMC (`TargetAMC` in the code) using `C-Target`.

Default settings:

```text
epochs          50
batch size      128
learning rate   1e-3
seed            2027
```

Run:

```bash
python 03_train_target_amc_0909.py
```

Main outputs:

```text
artifacts/models/target_amc_0909.pt
artifacts/logs/target_amc_training_0909.csv
```

The checkpoint records the source dataset, training SNRs, test accuracy, and the main training arguments.

---

### `04_train_surrogate_amc.py`

Trains the architecture-distinct `SurrogateAMC` using `C-Surrogate`. This clean-trained checkpoint is the common initialization for subsequent hardening experiments.

Default settings:

```text
epochs          100
batch size      256
learning rate   1e-3
seed            2028
```

Run:

```bash
python 04_train_surrogate_amc.py
```

The resulting model and training history are stored under `artifacts/models/` and `artifacts/logs/`.

---

### `02_train_diffusion_0909.py`

Trains the **clean class-conditional diffusion model** used by Diffusion-MMCD.

The model is trained only on legitimate `C-Target` signals and is conditioned on the modulation label. Random condition dropout is used during training.

Default settings:

```text
epochs                     200
batch size                 640
learning rate              2e-4
diffusion steps            500
condition-drop probability 0.1
seed                       2040
training SNRs              -10 to 30 dB in 5-dB steps
```

Run:

```bash
python 02_train_diffusion_0909.py
```

Main outputs:

```text
artifacts/models/clean_diffusion_target_label_only_0909.pt
artifacts/logs/clean_diffusion_target_label_only_0909.csv
```

The lowest validation-loss checkpoint is retained.

---

### `02_train_adversarial_diffusion_Uus_0909.py`

Trains the **source-guided adversarial diffusion generator**. The source AMC is frozen; gradients pass through it during training to update only the diffusion generator.

The current implementation uses a full differentiable DDPM reverse chain together with a bounded perturbation mapping and anti-saturation regularization.

Default settings:

```text
epochs                    15
train batch size          16
validation batch size     16
learning rate             1e-3
diffusion steps           10
seed                      2040
validation seed           3040
AMP                       disabled by default

attack objective          probability suppression
epsilon curriculum        0.30, 0.20, 0.10, 0.05, 0.03
validation epsilons       0.03, 0.05, 0.10, 0.20, 0.30

lambda_power              0.01
lambda_frequency          0.01
lambda_low_frequency      0.05
lambda_saturation         0.05

frequency keep ratio      0.25
low-frequency ratio       0.02
tanh temperature          2.0
squash limit              1.5
saturation threshold      0.98
```

Run:

```bash
python 02_train_adversarial_diffusion_Uus_0909.py
```

Main output:

```text
artifacts/models/adversarial_diffusion_unsupervised_0909.pt
```

The script uses conservative full-FP32 defaults for an approximately 8-GB-class GPU. CUDA AMP is optional:

```bash
python 02_train_adversarial_diffusion_Uus_0909.py --amp
```

---

### `05_offline_adversarial_augmentation_compare_0909_loss_retrain.py`

Runs the controlled **offline adversarial hardening** comparison.

Default experiments:

```text
clean                          Clean-FT
pgd                            PGD-OAT
diffusion_k1                   single-trajectory diffusion hardening
diffusion_multistart_cycle     multi-start diffusion diversity
diffusion_multistep_cycle      multi-depth DDIM diversity
diffusion_diverse_cycle        combined diversity (DD-OAT)
```

The final paper method corresponds to:

```text
diffusion_diverse_cycle
```

Default hardening settings:

```text
training epsilon          0.10
PGD steps                 10
multi-start K             4
DDIM reverse depths       2, 4, 6, 8
DDIM eta                  0.0

epochs                    50
training batch size       128
learning rate             1e-4
weight decay              1e-4
adversarial ratio         0.5
seed                      2040
```

The DDIM depths are **sampling depths**, not early-stopping timesteps. Each trajectory is propagated to the final perturbation. The depths `2, 4, 6, 8` are used jointly as diversity settings.

Checkpoint selection uses the fixed validation score:

```text
0.20 × clean validation accuracy
+ 0.40 × robustness to fixed source-PGD
+ 0.40 × robustness to fixed source-diffusion attacks
```

All selected defense models are trained for the full 50-epoch budget; early stopping is disabled. Diversity banks may contain multiple stored perturbations, but the cycle policy preserves a comparable number of optimization pairs per epoch.

Run all default experiments:

```bash
python 05_offline_adversarial_augmentation_compare_0909_loss_retrain.py
```

Run only the proposed DD-OAT experiment:

```bash
python 05_offline_adversarial_augmentation_compare_0909_loss_retrain.py     --experiments diffusion_diverse_cycle
```

Default output root:

```text
artifacts/offline_targetsource_to_surrogate_hardening_0909/
```

The script supports cached attack banks and automatic resume to avoid repeatedly generating expensive offline adversarial datasets.

> **Auxiliary scripts.** The hardening controller can call additional training/evaluation helper scripts. These auxiliary evaluation utilities and plotting scripts are being cleaned up and will be added/organized in subsequent repository updates. The core adversarial-bank construction, diversity scheduling, hardening logic, checkpoint metadata, and main training configurations are already provided.

---

## 4. Minimal Reproduction Workflow

From the repository root:

```bash
# 1. Generate the four independent A2G datasets
python 01_generate_clean_datasets_0909.py

# 2. Train the fixed source AMC
python 03_train_target_amc_0909.py

# 3. Train the clean surrogate AMC
python 04_train_surrogate_amc.py

# 4. Train the clean class-conditional diffusion detector
python 02_train_diffusion_0909.py

# 5. Train the source-guided adversarial diffusion generator
python 02_train_adversarial_diffusion_Uus_0909.py

# 6. Run diverse offline hardening
python 05_offline_adversarial_augmentation_compare_0909_loss_retrain.py     --experiments diffusion_diverse_cycle
```

Most scripts automatically select CUDA when available and otherwise fall back to CPU. GPU execution is strongly recommended for diffusion training and adversarial-bank generation.

---

## 5. Reproducibility Notes

### Random seeds

The pipeline uses fixed seeds at the data-generation and training stages. The four clean datasets use different role-specific seeds so that source-model, surrogate-model, diffusion-attack, and held-out evaluation data remain independently generated.

### Saved metadata

Where applicable, checkpoints record relevant training information such as:

- source dataset and pipeline version;
- random seed;
- optimizer/training arguments;
- diffusion steps;
- epsilon schedule;
- perturbation-loss weights;
- normalizer state;
- best-validation checkpoint information.

### Reference hardware

The scripts support CUDA automatically when available. The adversarial diffusion defaults were chosen conservatively for an approximately 8-GB GPU. The inference-time computational-cost measurements reported in the manuscript were obtained using an **NVIDIA GeForce RTX 4060 Laptop GPU** with batch-one FP32 inference.

### Software environment

The core implementation uses:

```text
Python
PyTorch
NumPy
pandas
scikit-learn
tqdm
```

Exact package versions and a requirements/environment file will be added during repository cleanup. Until then, the command-line defaults embedded in the released scripts should be treated as the reference configuration.

---

## 6. Artifact Layout

The pipeline writes outputs under `artifacts/`, typically:

```text
artifacts/
├── data/
│   ├── clean_diff_0909.pt
│   ├── clean_target_0909.pt
│   ├── clean_surrogate_0909.pt
│   └── clean_eval_0909.pt
├── models/
│   ├── target_amc_0909.pt
│   ├── surrogate_amc_0909.pt
│   ├── clean_diffusion_target_label_only_0909.pt
│   └── adversarial_diffusion_unsupervised_0909.pt
├── logs/
└── offline_targetsource_to_surrogate_hardening_0909/
```

Exact filenames of cached attack banks and hardened checkpoints contain experiment metadata such as epsilon, random seed, attack type, and diversity configuration.

---

## 7. Notes on the Current Release

This repository is being released alongside the manuscript revision. The **core algorithms, training logic, default hyperparameters, and main execution pipeline are already provided**. We will continue to organize and supplement the repository with:

- remaining evaluation utilities;
- publication plotting scripts;
- package-version lock files;
- additional documentation and example outputs.

If you encounter an inconsistency while reproducing an experiment, please open an issue and include the script name, command-line arguments, and generated checkpoint metadata.

---

## Citation

If this repository is useful for your research, please cite the associated paper. The final BibTeX entry will be added after publication.
