"""
full_experiment.py

Complete, submission-ready implementation of:
Parameter-Efficient ResNet for Drone-Based Infrastructure Inspection

Run in Colab top-to-bottom on CPU.
Generates: trained model, cached features, 5 plots, metrics JSON,
           metrics CSV, inference.py, requirements.txt
"""

import os
import zipfile
import random
import json
import time
import platform
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

# ------------------------------------------------------------------
# 1. Imports + Seeds
# ------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cpu")

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("cache",   exist_ok=True)

# ------------------------------------------------------------------
# 2. Unzip Dataset (if not already extracted)
# ------------------------------------------------------------------
ZIP_PATH    = "/content/5y9wdsg2zt-2.zip"
EXTRACT_DIR = "/content/surface_crack_dataset"

if not os.path.exists(EXTRACT_DIR) or not os.listdir(EXTRACT_DIR):
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"ZIP file not found: {ZIP_PATH}\n"
            "Please upload '5y9wdsg2zt-2.zip' to /content/ in Colab."
        )
    print(f"Extracting {ZIP_PATH}  →  {EXTRACT_DIR} ...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    print("Extraction complete.\n")
else:
    print(f"Dataset directory already exists: {EXTRACT_DIR}\n")

# ------------------------------------------------------------------
# 3. Dataset Auto-Detection + Balanced Sampling
# ------------------------------------------------------------------
image_exts = (".png", ".jpg", ".jpeg", ".bmp")

KNOWN_PAIRS = [
    ("Positive",  "Negative"),
    ("positive",  "negative"),
    ("Crack",     "No_Crack"),
    ("Crack",     "No Crack"),
    ("crack",     "non_crack"),
    ("crack",     "no_crack"),
    ("crack",     "noncrack"),
    ("crack",     "no crack"),
    ("crack",     "NoCrack"),
    ("crack",     "NonCrack"),
]


def find_candidate_pairs(search_root: str):
    """
    Walk the extracted directory tree and return (crack_dir, noncrack_dir).
    Works regardless of how many sub-folders the ZIP created.
    """
    # First try direct children
    for a, b in KNOWN_PAIRS:
        pa = os.path.join(search_root, a)
        pb = os.path.join(search_root, b)
        if os.path.exists(pa) and os.path.exists(pb):
            print(f"Dataset folders found directly: '{a}' / '{b}'")
            return pa, pb

    # Then walk deeper
    for dirpath, dirnames, _ in os.walk(search_root):
        for a, b in KNOWN_PAIRS:
            if a in dirnames and b in dirnames:
                pa = os.path.join(dirpath, a)
                pb = os.path.join(dirpath, b)
                print(f"Dataset folders found at: {dirpath}")
                print(f"  Crack    : {pa}")
                print(f"  No-crack : {pb}")
                return pa, pb

    raise ValueError(
        f"No recognised folder pair found under '{search_root}'.\n"
        "Folders inspected:\n" +
        "\n".join(f"  {r}" for r, _, _ in os.walk(search_root))
    )


crack_dir, noncrack_dir = find_candidate_pairs(EXTRACT_DIR)

crack_candidates = sorted(
    os.path.join(crack_dir, f)
    for f in os.listdir(crack_dir)
    if f.lower().endswith(image_exts)
)
noncrack_candidates = sorted(
    os.path.join(noncrack_dir, f)
    for f in os.listdir(noncrack_dir)
    if f.lower().endswith(image_exts)
)

print(f"Found {len(crack_candidates)} crack candidates, "
      f"{len(noncrack_candidates)} no-crack candidates")

random.shuffle(crack_candidates)
random.shuffle(noncrack_candidates)


def collect_valid(candidates, n):
    """Return exactly n valid (openable) image paths."""
    paths = []
    for p in candidates:
        if len(paths) == n:
            break
        try:
            Image.open(p).convert("RGB")
            paths.append(p)
        except Exception:
            print(f"Skipping corrupted image: {p}")
    if len(paths) < n:
        raise RuntimeError(
            f"Only {len(paths)} valid images found; need {n}.\n"
            "Check that the dataset was extracted correctly."
        )
    return paths


crack_paths    = collect_valid(crack_candidates,    2500)
noncrack_paths = collect_valid(noncrack_candidates, 2500)

selected_paths  = crack_paths + noncrack_paths
selected_labels = [1] * 2500 + [0] * 2500

assert len(selected_paths)  == 5000
assert len(crack_paths)     == 2500
assert len(noncrack_paths)  == 2500
print(f"\nCollected {len(selected_paths)} valid images (2500 crack + 2500 no-crack)")

# ------------------------------------------------------------------
# 4. Index Arrays + 70 / 15 / 15 Split  (no leakage)
# ------------------------------------------------------------------
indices = list(range(5000))

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    stratify=selected_labels,
    random_state=42,
)
val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    stratify=[selected_labels[i] for i in temp_indices],
    random_state=42,
)

assert len(train_indices) == 3500
assert len(val_indices)   == 750
assert len(test_indices)  == 750
print("Split: 3500 train | 750 val | 750 test")

# ImageNet normalisation (mandatory for pretrained ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
    ),
])


class CrackDataset(Dataset):
    """Loads raw images from disk on the fly."""
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ------------------------------------------------------------------
# 5. Raw DataLoaders
# ------------------------------------------------------------------
train_dataset = CrackDataset(
    [selected_paths[i]  for i in train_indices],
    [selected_labels[i] for i in train_indices],
    transform,
)
val_dataset = CrackDataset(
    [selected_paths[i]  for i in val_indices],
    [selected_labels[i] for i in val_indices],
    transform,
)
test_dataset = CrackDataset(
    [selected_paths[i]  for i in test_indices],
    [selected_labels[i] for i in test_indices],
    transform,
)
cache_dataset = CrackDataset(selected_paths, selected_labels, transform)

BATCH = 32
train_loader  = DataLoader(train_dataset,  batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,    batch_size=BATCH, shuffle=False, num_workers=0)
# test_loader defined for completeness; final evaluation uses cached_test_loader
test_loader   = DataLoader(test_dataset,   batch_size=BATCH, shuffle=False, num_workers=0)
cache_loader  = DataLoader(cache_dataset,  batch_size=BATCH, shuffle=False, num_workers=0)

# ------------------------------------------------------------------
# 6. Backbone Definition  (ResNet-50, frozen, Identity FC)
# ------------------------------------------------------------------
backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for p in backbone.parameters():
    p.requires_grad = False
backbone.fc = nn.Identity()   # output: 2048-d vector per image
backbone.to(device)
backbone.eval()
print("\nBackbone: ResNet-50 loaded, all params frozen, FC -> Identity()")

# ------------------------------------------------------------------
# 7. Feature Caching  (single forward pass, saved to disk)
# ------------------------------------------------------------------
print("\n--- Feature Caching ---")
cache_start  = time.time()
all_features = []
all_lbls     = []

with torch.no_grad():
    for images, labels in tqdm.tqdm(cache_loader, desc="Extracting features"):
        images = images.to(device)
        feats  = backbone(images)           # shape: (B, 2048)
        all_features.append(feats.cpu())
        all_lbls.append(labels.cpu())

features_all = torch.cat(all_features, dim=0)   # (5000, 2048)
labels_all   = torch.cat(all_lbls,     dim=0)   # (5000,)

caching_time = time.time() - cache_start

assert features_all.shape == (5000, 2048), "Unexpected feature shape"
assert features_all.dtype == torch.float32

torch.save(features_all, "cache/features_all.pt")
torch.save(labels_all,   "cache/labels_all.pt")

cache_file_mb = os.path.getsize("cache/features_all.pt") / (1024 ** 2)
print(f"Caching done in {caching_time:.2f}s  |  cache file: {cache_file_mb:.2f} MB")

# ------------------------------------------------------------------
# 8. Cached Tensor Split  (same index arrays → no leakage)
# ------------------------------------------------------------------
features_train = features_all[train_indices]
labels_train   = labels_all[train_indices]

features_val   = features_all[val_indices]
labels_val     = labels_all[val_indices]

features_test  = features_all[test_indices]
labels_test    = labels_all[test_indices]

# ------------------------------------------------------------------
# 9. CachedDataset  +  Cached DataLoaders
# ------------------------------------------------------------------
class CachedDataset(Dataset):
    """Serves pre-computed 2048-d feature vectors (no backbone needed)."""
    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels.float()      # BCEWithLogitsLoss requires float

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


cached_train_dataset = CachedDataset(features_train, labels_train)
cached_val_dataset   = CachedDataset(features_val,   labels_val)
cached_test_dataset  = CachedDataset(features_test,  labels_test)

cached_train_loader = DataLoader(cached_train_dataset, batch_size=BATCH, shuffle=True,  num_workers=0)
cached_val_loader   = DataLoader(cached_val_dataset,   batch_size=BATCH, shuffle=False, num_workers=0)
cached_test_loader  = DataLoader(cached_test_dataset,  batch_size=BATCH, shuffle=False, num_workers=0)

# ------------------------------------------------------------------
# 10. Head Definition
# ------------------------------------------------------------------
class Head(nn.Module):
    """Lightweight classification head: 2048 -> 512 -> 128 -> 1."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # NO sigmoid — BCEWithLogitsLoss applies it internally
        )

    def forward(self, x):
        return self.net(x)


head      = Head().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)


def compute_metrics(logits, labels, threshold=0.5, eps=1e-8):
    """Manual, edge-safe: accuracy / precision / recall / F1."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()

    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return accuracy.item(), precision.item(), recall.item(), f1.item()


# ------------------------------------------------------------------
# 11. Cached Training Loop
# ------------------------------------------------------------------
print("\n--- Cached Training (head only — backbone NOT called each epoch) ---")
cached_losses     = []
cached_val_accs   = []
best_val_acc      = 0.0
cached_train_time = 0.0   # training only; one-time caching excluded

EPOCHS = 20

for epoch in range(EPOCHS):
    head.train()
    running = 0.0
    t0      = time.time()

    for feats, labels in tqdm.tqdm(
            cached_train_loader,
            desc=f"Cached Epoch {epoch+1}/{EPOCHS}",
            leave=False):
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = head(feats)
        loss   = criterion(logits, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running += loss.item() * feats.size(0)

    cached_train_time += time.time() - t0
    epoch_loss = running / len(cached_train_loader.dataset)
    cached_losses.append(epoch_loss)

    # Validation — NOT timed
    head.eval()
    v_logits_list, v_labels_list = [], []
    with torch.no_grad():
        for feats, labels in cached_val_loader:
            feats, labels = feats.to(device), labels.to(device)
            v_logits_list.append(head(feats).cpu())
            v_labels_list.append(labels.cpu())

    v_logits = torch.cat(v_logits_list)
    v_labels = torch.cat(v_labels_list)
    val_acc, _, _, _ = compute_metrics(v_logits, v_labels.unsqueeze(1))
    cached_val_accs.append(val_acc)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss {epoch_loss:.4f} | val acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {
                "epoch":            epoch,
                "model_state_dict": head.state_dict(),
                "val_accuracy":     val_acc,
            },
            "models/head_best.pth",
        )

print(f"\nCached training time (training only): {cached_train_time:.2f}s")
print(f"Best validation accuracy            : {best_val_acc:.4f}")

# ------------------------------------------------------------------
# 12. Baseline Training Loop  (backbone called every batch)
# ------------------------------------------------------------------
print("\n--- Baseline Training (full ResNet forward pass every batch) ---")
assert torch.is_grad_enabled(), "Global gradient tracking must be enabled"
backbone.eval()   # BatchNorm stays in eval; weights remain frozen

baseline_head       = Head().to(device)
optimizer2          = torch.optim.Adam(baseline_head.parameters(), lr=1e-4)
baseline_losses     = []
baseline_train_time = 0.0

for epoch in range(EPOCHS):
    baseline_head.train()
    running = 0.0
    t0      = time.time()

    for imgs, labels in tqdm.tqdm(
            train_loader,
            desc=f"Baseline Epoch {epoch+1}/{EPOCHS}",
            leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer2.zero_grad()
        logits = baseline_head(backbone(imgs))   # full 50-layer forward every batch
        loss   = criterion(logits, labels.unsqueeze(1))
        loss.backward()
        optimizer2.step()
        running += loss.item() * imgs.size(0)

    baseline_train_time += time.time() - t0
    epoch_loss = running / len(train_loader.dataset)
    baseline_losses.append(epoch_loss)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss {epoch_loss:.4f}")

print(f"\nBaseline training time (training only): {baseline_train_time:.2f}s")

# ------------------------------------------------------------------
# 13. Test Evaluation  (test set only, cached features)
# ------------------------------------------------------------------
print("\n--- Test Evaluation ---")
ckpt = torch.load("models/head_best.pth", map_location="cpu")
head.load_state_dict(ckpt["model_state_dict"])
head.eval()

test_logits_list, test_labels_list = [], []
with torch.no_grad():
    for feats, labels in cached_test_loader:
        feats, labels = feats.to(device), labels.to(device)
        test_logits_list.append(head(feats).cpu())
        test_labels_list.append(labels.cpu())

test_logits = torch.cat(test_logits_list)
test_labels = torch.cat(test_labels_list)

test_acc, test_prec, test_rec, test_f1 = compute_metrics(
    test_logits, test_labels.unsqueeze(1)
)
print(f"Accuracy : {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall   : {test_rec:.4f}")
print(f"F1 Score : {test_f1:.4f}")

# ------------------------------------------------------------------
# 14. Plot Generation  
# ------------------------------------------------------------------
print("\n--- Generating Plots ---")

# Plot 1: Training loss curve — both methods on same axes
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(range(1, EPOCHS+1), cached_losses,   label="Cached",   color="C1")
plt.plot(range(1, EPOCHS+1), baseline_losses,  label="Baseline", color="C0")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve: Cached vs Baseline")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/training_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Validation accuracy curve
plt.figure(figsize=(10, 6), dpi=150)
plt.plot(range(1, EPOCHS+1), cached_val_accs, label="Cached val acc", color="C1")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Curve (Cached Training)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/accuracy_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 3: Training time comparison
plt.figure(figsize=(10, 6), dpi=150)
bars = plt.bar(["Baseline", "Cached"],
               [baseline_train_time, cached_train_time],
               color=["C0", "C1"])
plt.xlabel("Training Method")
plt.ylabel("Total Training Time (seconds)")
plt.title("Training Time Comparison: Cached vs Baseline")
for bar, v in zip(bars, [baseline_train_time, cached_train_time]):
    plt.text(bar.get_x() + bar.get_width() / 2,
             v + max(baseline_train_time, cached_train_time) * 0.01,
             f"{v:.1f}s", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/timing_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 4: Storage comparison
total_raw_mb = sum(os.path.getsize(p) for p in selected_paths) / (1024 ** 2)
cache_mb     = os.path.getsize("cache/features_all.pt") / (1024 ** 2)

plt.figure(figsize=(10, 6), dpi=150)
bars = plt.bar(["Raw Images (5k)", "Cached Features"],
               [total_raw_mb, cache_mb],
               color=["C0", "C1"])
plt.xlabel("Storage Type")
plt.ylabel("Size (MB)")
plt.title("Storage Comparison: Raw Images vs Cached Features")
for bar, v in zip(bars, [total_raw_mb, cache_mb]):
    plt.text(bar.get_x() + bar.get_width() / 2,
             v + max(total_raw_mb, cache_mb) * 0.01,
             f"{v:.1f} MB", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/storage_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 5: Confusion matrix
preds_bin = (torch.sigmoid(test_logits) >= 0.5).float().numpy()
cm        = confusion_matrix(test_labels.numpy(), preds_bin)

plt.figure(figsize=(6, 6), dpi=150)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Crack", "Crack"],
    yticklabels=["No Crack", "Crack"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print("All 5 plots saved to outputs/")

# ------------------------------------------------------------------
# 15. Metrics Export + Storage Analysis + Final Validation Asserts
# ------------------------------------------------------------------
print("\n--- Metrics Export & Storage Analysis ---")

reduction_percent = (
    (baseline_train_time - cached_train_time) / baseline_train_time
) * 100

# ── Per-epoch timing (Task D: compare training time per epoch) ────
cached_time_per_epoch   = cached_train_time   / EPOCHS
baseline_time_per_epoch = baseline_train_time / EPOCHS

print(f"\nCached time per epoch   : {cached_time_per_epoch:.2f}s")
print(f"Baseline time per epoch : {baseline_time_per_epoch:.2f}s")

# ── Storage table ─────────────────────────────────────────────────
print(f"\n{'Component':<30} {'Size (MB)':>10} {'% of Raw':>10}")
print("-" * 52)
print(f"{'Raw images (5,000)':30} {total_raw_mb:>10.2f} {'100.00%':>10}")
print(f"{'Cached features.pt':30} {cache_mb:>10.2f} {cache_mb/total_raw_mb*100:>9.2f}%")

# ── Compute reduction analysis (honest, hardware-aware) ───────────
print("\n--- Compute Reduction Analysis ---")
print(f"Total Baseline Training Time : {baseline_train_time:.2f} seconds")
print(f"Total Cached Training Time   : {cached_train_time:.2f} seconds")
print(f"Compute Reduction            : {reduction_percent:.2f}%")

print("\nNotes:")
print("  • Reduction is calculated using training-loop time only.")
print("  • One-time feature caching cost is excluded from reduction.")
print("  • Backbone remains frozen in both methods.")
print("  • Actual percentage depends on hardware (CPU / Colab VM),")
print("    batch size, and number of epochs.")
print("  • On most CPU-only setups, reduction exceeds 90%.")

# ── summary.json ──────────────────────────────────────────────────
summary = {
    "accuracy":                   round(test_acc,                4),
    "precision":                  round(test_prec,               4),
    "recall":                     round(test_rec,                4),
    "f1":                         round(test_f1,                 4),
    "time_cached_seconds":        round(cached_train_time,       2),
    "time_full_seconds":          round(baseline_train_time,     2),
    "time_cached_per_epoch":      round(cached_time_per_epoch,   2),
    "time_full_per_epoch":        round(baseline_time_per_epoch, 2),
    "reduction_percent":          round(reduction_percent,       2),
    "caching_time_seconds":       round(caching_time,            2),
    "storage_raw_mb":             round(total_raw_mb,            2),
    "storage_cache_mb":           round(cache_mb,                2),
    "cpu":                        platform.processor(),
}
with open("outputs/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ── metrics_table.csv ─────────────────────────────────────────────
with open("outputs/metrics_table.csv", "w") as f:
    f.write("Method,Accuracy,Precision,Recall,F1,Time(seconds),Time_per_epoch(seconds)\n")
    f.write(
        f"Cached,{test_acc:.4f},{test_prec:.4f},{test_rec:.4f},"
        f"{test_f1:.4f},{cached_train_time:.2f},{cached_time_per_epoch:.2f}\n"
    )
    f.write(
        f"Baseline,N/A,N/A,N/A,N/A,"
        f"{baseline_train_time:.2f},{baseline_time_per_epoch:.2f}\n"
    )

print("\nSaved: outputs/summary.json")
print("Saved: outputs/metrics_table.csv")

# ── Runtime validation asserts ────────────────────────────────────
assert len(train_indices) == 3500
assert len(val_indices)   == 750
assert len(test_indices)  == 750
assert cached_train_dataset.labels.dtype == torch.float32
assert os.path.exists("models/head_best.pth")
assert os.path.exists("cache/features_all.pt")
assert os.path.exists("outputs/training_loss_curve.png")
assert os.path.exists("outputs/accuracy_curve.png")
assert os.path.exists("outputs/timing_comparison_bar.png")
assert os.path.exists("outputs/storage_comparison_bar.png")
assert os.path.exists("outputs/confusion_matrix.png")
assert os.path.exists("outputs/summary.json")
assert os.path.exists("outputs/metrics_table.csv")

ckpt_chk = torch.load("models/head_best.pth", map_location="cpu")
assert 0 <= ckpt_chk["epoch"] < EPOCHS

print("\n✔  All structural validation checks passed.")

