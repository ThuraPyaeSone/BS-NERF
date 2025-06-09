import open3d as o3d
import os
import numpy as np

# === Paths ===
pcd_path = "/home/thurapyaesone/Desktop/nerf2/nerfstudio-method-semanticnerf/exports/pcd/point_cloud.ply"
backproj_mask_path = "/home/thurapyaesone/Desktop/nerf2/backprojected_mask_points.ply"
cropped_output_path = "/home/thurapyaesone/Desktop/nerf2/nerfstudio-method-semanticnerf/exports/pcd/point_cloud_mask_region_expanded.ply"

# === Load Full Point Cloud ===
if not os.path.exists(pcd_path):
    raise FileNotFoundError(f"Point cloud not found: {pcd_path}")
pcd = o3d.io.read_point_cloud(pcd_path)
print(f"[INFO] Loaded full point cloud: {len(pcd.points)} points")

# === Load Backprojected Mask ===
if not os.path.exists(backproj_mask_path):
    raise FileNotFoundError(f"Backprojected mask not found: {backproj_mask_path}")
mask_pcd = o3d.io.read_point_cloud(backproj_mask_path)
mask_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for visibility
print(f"[INFO] Loaded backprojected mask: {len(mask_pcd.points)} points")

# === Get and expand bounding box
bbox = mask_pcd.get_axis_aligned_bounding_box()
margin_scale = 5  # Expand by 20%
center = bbox.get_center()
extent = bbox.get_extent() * margin_scale
expanded_bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=center - extent / 2,
    max_bound=center + extent / 2
)

# === Crop full point cloud using expanded box
cropped_pcd = pcd.crop(expanded_bbox)
print(f"[INFO] Cropped expanded region: {len(cropped_pcd.points)} points")

# === Save the cropped region
o3d.io.write_point_cloud(cropped_output_path, cropped_pcd)
print(f"[INFO] Saved cropped region to:\n  {cropped_output_path}")

# === Optional: Visualize result
o3d.visualization.draw_geometries(
    [cropped_pcd, mask_pcd],
    window_name="Expanded Crop + Backprojected Mask",
    width=1280,
    height=720
)
