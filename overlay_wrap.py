import cv2
import os
import numpy as np
from tqdm import tqdm

# === Paths ===
rendered_frames_dir = "/home/thurapyaesone/Desktop/nerf2/data/Knee/rendered_frames"
warped_masks_dir = "/home/thurapyaesone/Desktop/nerf2/data/Knee/warped_masks"
output_video_path = "/home/thurapyaesone/Desktop/nerf2/data/Knee/warped_overlay_video.mp4"

# === Collect sorted frame list ===
frames = sorted(f for f in os.listdir(rendered_frames_dir) if f.endswith(".png"))

# === Read video properties from first frame ===
first_frame = cv2.imread(os.path.join(rendered_frames_dir, frames[0]))
h, w, _ = first_frame.shape
fps = 30  # or match original video if known

# === Video writer setup ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

# === Overlay and write video ===
for fname in tqdm(frames):
    rendered_path = os.path.join(rendered_frames_dir, fname)
    mask_path = os.path.join(warped_masks_dir, fname)

    rendered = cv2.imread(rendered_path)
    if not os.path.exists(mask_path):
        out.write(rendered)
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # Resize mask to match rendered image if needed
    if color_mask.shape != rendered.shape:
        color_mask = cv2.resize(color_mask, (rendered.shape[1], rendered.shape[0]))

    blended = cv2.addWeighted(rendered, 1.0, color_mask, 0.4, 0)
    out.write(blended)

out.release()
print(f" Overlay video saved: {output_video_path}")