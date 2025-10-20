#!/usr/bin/env python3
"""
pca_visualize_patches.py

Visualize patch embeddings from different frozen backbones:
- DinoBloom
- BLIP base / BLIP finetuned
- MedGemma (vision tower)
- ResNet50

Method:
1. Extract patch embeddings (per-image patch tokens, not CLS).
2. Apply PCA -> reduce to 3D.
3. Map PCA components to RGB.
4. Reshape into patch grid, upscale to image size.
5. Save side-by-side comparisons with the input.

"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.utils import save_image
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt

from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModel, AutoImageProcessor
import dinov2.models.vision_transformer as vits

# ========================
# CONFIG
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = "./pca_patch_viz"
os.makedirs(out_dir, exist_ok=True)

# Choose test images
test_images = [
    "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/cropped_imgs_desc/basophil_BA_752996.jpg",
    "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/cropped_imgs_desc/train_2427_30_24_1000_CML.png",
    "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/OOD_test/golden-retriever-tongue-out.jpg",
    "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/ExternalImages/MO_350301.jpg",
    "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/ExternalImages/test_545_28_32_1000_CML.png",
    "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/ExternalImages/page_38_img_7_leukemia.jpg",
]

# Backbone settings
backbones = [
    {"name": "dinobloom_b", "ckpt": "./DinoBloom-B.pth", "arch": "base"},
    {"name": "blip_base", "path": "Salesforce/blip-image-captioning-base"},
    {"name": "blip_ft", "path": "./EINDRESULTAAT_BLIP1_fullyfinetuned"},
    {"name": "medgemma", "path": "google/medgemma-4b-it"},
    {"name": "resnet50", "arch": "resnet50"},
]

# Common transforms
img_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

# ========================
# HELPERS
# ========================

def get_patch_tokens(hidden_state: torch.Tensor):
    """
    Ensures we return only patch tokens (no CLS).
    hidden_state: (B, N, D)
    """
    B, N, D = hidden_state.shape
    # check if N is already a perfect square
    if int(np.sqrt(N))**2 == N:
        return hidden_state  # already patch tokens
    elif int(np.sqrt(N-1))**2 == (N-1):
        return hidden_state[:, 1:, :]  # drop CLS
    else:
        raise ValueError(f"Unexpected token count {N}, not matching grid.")

def to_rgb_from_pca(patches: torch.Tensor):
    """
    patches: (num_patches, hidden_dim)
    """
    n = patches.size(0)
    patch_hw = int(np.sqrt(n))
    if patch_hw * patch_hw != n:
        raise ValueError(f"Num patches {n} is not a perfect square, cannot reshape to grid.")

    pca = PCA(n_components=3)
    X = patches.cpu().numpy()
    Xp = pca.fit_transform(X)
    Xp -= Xp.min(0, keepdims=True)
    Xp /= Xp.max(0, keepdims=True) + 1e-9
    Xp = (Xp * 255).astype(np.uint8)

    img_rgb = Xp.reshape(patch_hw, patch_hw, 3)
    img_rgb = Image.fromarray(img_rgb).resize((224, 224), resample=Image.NEAREST)
    return img_rgb


def save_side_by_side(input_img, viz_imgs, labels, save_path):
    n = len(viz_imgs) + 1
    plt.figure(figsize=(3*n, 3))
    plt.subplot(1, n, 1)
    plt.imshow(input_img)
    plt.title("Input")
    plt.axis("off")

    for i, (viz, lbl) in enumerate(zip(viz_imgs, labels), 2):
        plt.subplot(1, n, i)
        plt.imshow(viz)
        plt.title(lbl)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ========================
# EXTRACTORS
# ========================
class DinoBloomExtractor(nn.Module):
    def __init__(self, ckpt_path, arch="base"):
        super().__init__()
        if arch == "small":
            self.model = vits.vit_small(patch_size=14)
            self.out_dim = 384
        elif arch == "base":
            self.model = vits.vit_base(patch_size=14)
            self.out_dim = 768
        elif arch == "large":
            self.model = vits.vit_large(patch_size=14)
            self.out_dim = 1024
        elif arch == "giant":
            self.model = vits.vit_giant2(patch_size=14)
            self.out_dim = 1536
        state_dict = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(device)
    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        # returns a list of activations from chosen layers
        feats = self.model.get_intermediate_layers(img, n=1)[0]  # (B, N, D) possibly with CLS
        toks = get_patch_tokens(feats)  
        return toks

class BlipExtractor(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.processor = BlipProcessor.from_pretrained(path)
        m = BlipForConditionalGeneration.from_pretrained(path)
        self.vision = m.vision_model.to(device).eval()
    @torch.no_grad()
    def forward(self, pil_img):
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel = inputs["pixel_values"].to(device)
        out = self.vision(pixel_values=pixel)
        return out.last_hidden_state[:, 1:, :]  # drop CLS

class MedGemmaExtractor(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True,
                                               device_map="auto", torch_dtype=torch.float16)
        self.vision = getattr(self.model, "vision_tower", None) or getattr(self.model, "vision_model", None)
        self.vision.eval().to(device)
        self.processor = AutoImageProcessor.from_pretrained(path, trust_remote_code=True)
    @torch.no_grad()
    def forward(self, pil_img):
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel = inputs["pixel_values"].to(device)
        out = self.vision(pixel_values=pixel)
        patch_tokens = get_patch_tokens(out.last_hidden_state)
        return patch_tokens

class ResNetExtractor(nn.Module):
    def __init__(self, arch="resnet50"):
        super().__init__()
        self.model = getattr(models, arch)(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2]).to(device).eval()
    @torch.no_grad()
    def forward(self, img_tensor: torch.Tensor):
        f = self.model(img_tensor)  # (B, C, H, W)
        B, C, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(B, H*W, C)
        return f

# ========================
# MAIN
# ========================
def main():
    for img_path in test_images:
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = img_transform(pil_img).unsqueeze(0).to(device)

        viz_list, labels = [], []
        for bb in backbones:
            try:
                if bb["name"].startswith("dinobloom"):
                    extractor = DinoBloomExtractor(bb["ckpt"], arch=bb["arch"])
                    toks = extractor(img_tensor)[0]  # (n_patches, dim)
                    patch_hw = int(np.sqrt(toks.size(0)))
                    rgb = to_rgb_from_pca(toks)

                elif "blip" in bb["name"]:
                    extractor = BlipExtractor(bb["path"])
                    toks = extractor(pil_img)[0]
                    patch_hw = int(np.sqrt(toks.size(0)))
                    rgb = to_rgb_from_pca(toks)

                elif "medgemma" in bb["name"]:
                    extractor = MedGemmaExtractor(bb["path"])
                    toks = extractor(pil_img)[0]
                    patch_hw = int(np.sqrt(toks.size(0)))
                    rgb = to_rgb_from_pca(toks)

                elif "resnet" in bb["name"]:
                    extractor = ResNetExtractor(bb["arch"])
                    toks = extractor(img_tensor)[0]  # (n_patches, dim)
                    patch_hw = int(np.sqrt(toks.size(0)))
                    rgb = to_rgb_from_pca(toks)

                viz_list.append(rgb)
                labels.append(bb["name"])
                print(f"[OK] {bb['name']} on {img_path}")

            except Exception as e:
                print(f"[FAIL] {bb['name']}: {e}")

        save_path = os.path.join(out_dir, f"pca_patches_{os.path.basename(img_path)}.png")
        save_side_by_side(pil_img, viz_list, labels, save_path)
        print(f"Saved visualization -> {save_path}")


if __name__ == "__main__":
    main()
