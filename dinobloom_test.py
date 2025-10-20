import torch
from PIL import Image
from torchvision import transforms
import dinov2.models.vision_transformer as vits  # this is where the models live

# 1. Load a DinoBloom checkpoint (choose S, B, L, or G)
ckpt_path = "DinoBloom-B.pth"  # path to the checkpoint you downloaded
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Initialize the backbone (Base corresponds to DinoBloom-B)
model = vits.vit_base(patch_size=14)  # you can also try vit_small, vit_large, vit_giant2
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict, strict=False)  # strict=False allows for slight key mismatches
model.eval().to(device)

# 3. Image preprocessing (same as in paper/code)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# 4. Load your own cell image
img_path = "/cs/student/projects1/aibh/2024/jvanlogt/miniconda3/dissertation_code/Finetune_BLIP/data/sanity_check_images/train_2694_22_11_1000_AML.png"
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

# 5. Extract features
with torch.no_grad():
    # forward returns a tuple: (cls_token, patch_tokens)
    cls_output = model(x)  # typically use the CLS token as embedding
    features = cls_output

print("Feature vector shape:", features.shape)
print("First 5 dims:", features[0, :5].cpu().numpy())
