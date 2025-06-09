import open3d as o3d
import os
import numpy as np

# === Paths ===
pcd_path = "/home/thurapyaesone/Desktop/nerf2/nerfstudio-method-semanticnerf/exports/pcd/point_cloud.ply"
backproj_mask_path = "/home/thurapyaesone/Desktop/nerf2/backprojected_mask_points.ply"
cropped_output_path = "/home/thurapyaesone/Desktop/nerf2/nerfstudio-method-semanticnerf/exports/pcd/point_cloud_mask_region_transformed.ply"
transformed_mask_path = "/home/thurapyaesone/Desktop/nerf2/backprojected_mask_transformed.ply"
screenshot_path = "/home/thurapyaesone/Desktop/nerf2/scene_screenshot.png"

# === Load full point cloud
if not os.path.exists(pcd_path):
    raise FileNotFoundError(f"Point cloud not found: {pcd_path}")
pcd = o3d.io.read_point_cloud(pcd_path)
print(f"[INFO] Loaded full point cloud with {len(pcd.points)} points")

# === Load backprojected mask
if not os.path.exists(backproj_mask_path):
    raise FileNotFoundError(f"Backprojected mask not found: {backproj_mask_path}")
mask_pcd = o3d.io.read_point_cloud(backproj_mask_path)
print(f"[INFO] Loaded backprojected mask with {len(mask_pcd.points)} points")

# === Set mask color
mask_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red

# === Define rotation angles (in degrees)
angle_x = -78
angle_y = 48
angle_z = 140

# === Define scale factor (uniform scaling)
scale = 0.42

# === Convert to radians
rx = np.deg2rad(angle_x)
ry = np.deg2rad(angle_y)
rz = np.deg2rad(angle_z)

# === Rotation matrices
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(rx), -np.sin(rx)],
    [0, np.sin(rx), np.cos(rx)]
])
Ry = np.array([
    [np.cos(ry), 0, np.sin(ry)],
    [0, 1, 0],
    [-np.sin(ry), 0, np.cos(ry)]
])
Rz = np.array([
    [np.cos(rz), -np.sin(rz), 0],
    [np.sin(rz), np.cos(rz), 0],
    [0, 0, 1]
])

# === Combined rotation
R = Rz @ Ry @ Rx

# === Apply scaling to rotation matrix
R_scaled = scale * R

# === Translation vector
t = np.array([-4, -1.55, -1.5])

# === Build 4x4 transformation matrix
T = np.eye(4)
T[:3, :3] = R_scaled
T[:3, 3] = t
print("[INFO] Applied transform (with scaling):\n", T)

# === Transform the mask point cloud
mask_pcd.transform(T)

# === Compute and expand bounding box
bbox = mask_pcd.get_axis_aligned_bounding_box()
margin_scale = 20.0
center = bbox.get_center()
extent = bbox.get_extent() * margin_scale
expanded_bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=center - extent / 2,
    max_bound=center + extent / 2
)

# === Crop full point cloud
cropped_pcd = pcd.crop(expanded_bbox)
print(f"[INFO] Cropped region: {len(cropped_pcd.points)} points")
o3d.io.write_point_cloud(cropped_output_path, cropped_pcd)
print(f"[INFO] Saved cropped point cloud to:\n  {cropped_output_path}")

# === Save transformed mask
o3d.io.write_point_cloud(transformed_mask_path, mask_pcd)
print(f"[INFO] Saved transformed mask to:\n  {transformed_mask_path}")

# === Save screenshot of the combined scene
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vis.add_geometry(cropped_pcd)
vis.add_geometry(mask_pcd)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image(screenshot_path)
vis.destroy_window()
print(f"[INFO] Saved scene screenshot to:\n  {screenshot_path}")

# === Visualize interactively
o3d.visualization.draw_geometries(
    [cropped_pcd, mask_pcd],
    window_name="Scaled + Rotated Mask + Cropped Point Cloud",
    width=1280,
    height=720
)