# === inference_segmentation.py ===
import os
import cv2
import torch
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.transforms as T

# ---- CONFIG ----
IMAGE_DIR = "data/Knee/images"
OUTPUT_MASK_DIR = "data/Knee/inferred_masks"
OUTPUT_PREVIEW_DIR = "data/Knee/inferred_previews"
MODEL_PATH = "knee_segment_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
os.makedirs(OUTPUT_PREVIEW_DIR, exist_ok=True)

# ---- Load Model ----
model = deeplabv3_resnet50(pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---- Transforms ----
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor()
])

# ---- Inference ----
def overlay_mask(image, mask, alpha=0.5):
    overlay = image.copy()
    overlay[mask == 1] = (255, 0, 0)
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

for filename in sorted(os.listdir(IMAGE_DIR)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"[SKIP] Failed to load {filename}")
        continue

    image_resized = cv2.resize(image, (256, 256))
    input_tensor = transform(image_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)['out']
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    binary_mask = (pred == 1).astype(np.uint8) * 255
    binary_mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

    # Save mask
    mask_path = os.path.join(OUTPUT_MASK_DIR, filename)
    cv2.imwrite(mask_path, binary_mask_resized)

    # Save preview
    overlay = overlay_mask(image, binary_mask_resized // 255)
    preview_path = os.path.join(OUTPUT_PREVIEW_DIR, filename)
    cv2.imwrite(preview_path, overlay)

    print(f"[SAVED] {filename} → mask + preview")

print("[DONE] Inference complete.")
