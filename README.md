ResNet_SCD
Parameter-Efficient Surface Crack Detection using ResNet-50
 Overview

ResNet_SCD implements a compute-efficient deep learning pipeline for detecting structural cracks in concrete surfaces for drone-based infrastructure inspection.

The core contribution is a Feature Caching Protocol that reduces training compute by over 90% compared to standard baseline training, while maintaining strong classification performance.

The system is designed for CPU-only execution, reproducibility, and edge deployment feasibility.

 Core Idea

Instead of repeatedly executing the full ResNet-50 backbone during every training epoch:

The pretrained backbone is frozen

A single forward pass extracts 2048-dimensional feature vectors

Features are cached to disk

A lightweight classification head is trained on cached embeddings

This removes redundant convolutional computation and significantly reduces training time.

🏗 Architecture

Backbone: ResNet-50 (ImageNet pretrained, frozen)
Feature Vector: 2048 dimensions

Classification head:

𝐻
(
𝑥
)
=
𝜎
(
𝑊
3
⋅
ReLU
(
𝑊
2
⋅
ReLU
(
𝑊
1
⋅
𝑥
)
)
)
H(x)=σ(W
3
	​

⋅ReLU(W
2
	​

⋅ReLU(W
1
	​

⋅x)))

Layer structure:

2048 → 512 → 128 → 1

Loss: BCEWithLogitsLoss
Optimizer: Adam

📊 Evaluation

Reported on the held-out test set:

Accuracy

Precision

Recall

F1 Score

Total training time (baseline vs cached)

Per-epoch time comparison

Compute reduction percentage

Storage overhead analysis

Detailed metrics are saved in:

outputs/summary.json
outputs/metrics_table.csv
📂 Repository Structure
ResNet_SCD/
│
├── src/
│   ├── full_experiment.py
│   └── inference.py
│
├── models/
├── outputs/
├── cache/
├── report/
│   └── Final_Report.pdf
│
├── requirements.txt
├── README.md
└── .gitignore
⚙ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/ResNet_SCD.git
cd ResNet_SCD

Install dependencies:

pip install -r requirements.txt
▶ Training

From the project root:

python src/full_experiment.py

This generates:

Trained model → models/head_best.pth

Cached feature tensors → cache/

Plots and metrics → outputs/

🔎 Inference

After training:

python src/inference.py --image path/to/image.jpg

Example output:

Label      : Damaged
Probability: 0.9234
📚 Dataset

Mendeley Surface Crack Detection Dataset

Balanced subset of 5,000 images:

2,500 Crack

2,500 No-Crack

Split:

70% Training

15% Validation

15% Test

🎓 Academic Context

Course: Applied Computer Vision
Program: B.Tech (AIML)

This project demonstrates parameter-efficient training, compute reduction analysis, and storage-performance trade-off evaluation under CPU-only constraints.

Author

Sanjana Sameera
