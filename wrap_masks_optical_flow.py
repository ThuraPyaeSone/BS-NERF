import os
import cv2
import numpy as np
from tqdm import tqdm

# === Paths ===
image_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/images"
rendered_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/rendered_frames"
mask_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/masks"
output_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/warped_masks_colored"
os.makedirs(output_folder, exist_ok=True)

# === Sort and pair frames ===
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
rendered_files = sorted([f for f in os.listdir(rendered_folder) if f.endswith(".png")])

def warp_mask_using_optical_flow(src_img, tgt_img, src_mask):
    # Resize target image to match source
    tgt_img = cv2.resize(tgt_img, (src_img.shape[1], src_img.shape[0]))

    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        src_gray, tgt_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    h, w = flow.shape[:2]
    flow_map = np.zeros_like(flow)
    flow_map[..., 0] = np.arange(w)
    flow_map[..., 1] = np.arange(h)[:, np.newaxis]
    warped_coords = (flow_map + flow).astype(np.float32)

    warped_mask = cv2.remap(src_mask, warped_coords[..., 0], warped_coords[..., 1], interpolation=cv2.INTER_NEAREST)
    return warped_mask

# === Warp and save colored masks ===
for idx, rendered_name in tqdm(enumerate(rendered_files), total=len(rendered_files)):
    if idx >= len(image_files):
        src_name = image_files[-1]
    else:
        src_name = image_files[idx]

    src_img = cv2.imread(os.path.join(image_folder, src_name))
    tgt_img = cv2.imread(os.path.join(rendered_folder, rendered_name))
    src_mask_path = os.path.join(mask_folder, src_name)

    if not os.path.exists(src_mask_path):
        print(f"[Skip] Missing mask: {src_mask_path}")
        continue

    src_mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
    warped_mask = warp_mask_using_optical_flow(src_img, tgt_img, src_mask)

    # === Colorize and intensify the warped mask
    color_mask = cv2.applyColorMap(warped_mask, cv2.COLORMAP_JET)
    color_mask = cv2.convertScaleAbs(color_mask, alpha=1.5, beta=0)  # Increase intensity

    # === Save the colorized mask
    output_path = os.path.join(output_folder, rendered_name)
    cv2.imwrite(output_path, color_mask)

print(" All masks warped, colorized, and saved with higher visual density.")
