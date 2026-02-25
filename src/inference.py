# ------------------------------------------------------------------
# Inference helper (used internally + optional CLI)
# ------------------------------------------------------------------
def infer_image(image_path: str):
    """Run a single raw image through frozen backbone + best trained head."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img    = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = backbone(tensor)
        logit = head(feats)
        prob  = torch.sigmoid(logit).item()

    label = "Damaged" if prob >= 0.5 else "Safe"
    print(f"Label      : {label}")
    print(f"Probability: {prob:.4f}")
    return label, prob


# ------------------------------------------------------------------
# Generate inference.py
# ------------------------------------------------------------------
inference_code = '''\
"""
inference.py

Standalone inference script — drone infrastructure inspection.

Usage:
    python inference.py --image path/to/image.jpg

Output:
    Label      : Safe | Damaged
    Probability: float in [0, 1]
"""
import os
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

device = torch.device("cpu")

# Reconstruct backbone — must match training exactly
backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
backbone.fc = nn.Identity()
for p in backbone.parameters():
    p.requires_grad = False
backbone.eval()
backbone.to(device)


class Head(nn.Module):
    """Architecture must match full_experiment.py exactly."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


head = Head().to(device)

CHECKPOINT = "models/head_best.pth"
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}\\n"
        "Run full_experiment.py first to train and save the model."
    )

ckpt = torch.load(CHECKPOINT, map_location="cpu")
head.load_state_dict(ckpt["model_state_dict"])
head.eval()

# Preprocessing — must be identical to training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
    ),
])


def predict(image_path: str):
    """
    Predict whether an infrastructure image shows structural damage.

    Args:
        image_path (str): Path to JPEG or PNG image.

    Returns:
        label (str)  : "Safe" or "Damaged"
        prob  (float): damage probability in [0, 1]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot open image: {image_path}") from exc

    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = backbone(tensor)      # 2048-d feature vector
        logit = head(feats)
        prob  = torch.sigmoid(logit).item()

    label = "Damaged" if prob >= 0.5 else "Safe"
    return label, prob


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Drone Infrastructure Inspection — Inference"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image (JPEG / PNG)"
    )
    args = parser.parse_args()

    label, prob = predict(args.image)
    print(f"Label      : {label}")
    print(f"Probability: {prob:.4f}")
'''

with open("inference.py", "w") as f:
    f.write(inference_code)
print("inference.py written.")

