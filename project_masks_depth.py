import json
import numpy as np
import cv2
import os
import re
import matplotlib
matplotlib.use('Agg')  # Use headless backend for saving plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import open3d as o3d

def load_training_cameras_and_masks(transforms_json_path, mask_folder):
    with open(transforms_json_path, 'r') as f:
        transforms = json.load(f)

    intrinsics = {
        'fl_x': transforms['fl_x'],
        'fl_y': transforms['fl_y'],
        'cx': transforms['cx'],
        'cy': transforms['cy'],
        'w': transforms['w'],
        'h': transforms['h'],
    }

    cameras = []
    masks = []
    filenames = []
    for frame in transforms['frames']:
        pose = np.array(frame['transform_matrix'])
        file_path = frame['file_path']
        base_filename = os.path.basename(file_path)
        if not base_filename.endswith('.png'):
            base_filename += '.png'
        mask_path = os.path.join(mask_folder, base_filename)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if np.sum(mask > 0) == 0:
                print(f"[Warning] Empty mask: {mask_path}")
                continue
            cameras.append((pose, intrinsics))
            masks.append(mask)
            filenames.append(base_filename)
        else:
            print(f"[Warning] Mask not found: {mask_path}")

    print(f"[INFO] Loaded {len(masks)} masks and poses")
    return cameras, masks, filenames

def load_pfm(file_path, normalize=True, scale_factor=5.0):
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        color = header == 'PF'
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if not dim_match:
            raise Exception('Malformed PFM header.')
        width, height = map(int, dim_match.groups())
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

    min_val = np.min(data)
    max_val = np.max(data)
    print(f"[DEBUG] Depth stats: min={min_val:.4f}, max={max_val:.4f}, scale_factor={scale_factor}")

    if normalize:
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
        else:
            data[:] = 0

    data *= scale_factor
    return data

def backproject_mask_to_world_points(mask, depth_map, pose, intrinsics, subsample_rate=10):
    fx = intrinsics['fl_x']
    fy = intrinsics['fl_y']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return np.empty((0, 3))

    if subsample_rate > 1 and len(xs) > subsample_rate:
        idx = np.random.choice(len(xs), len(xs) // subsample_rate, replace=False)
        xs = xs[idx]
        ys = ys[idx]

    depths = depth_map[ys, xs]
    x_cam = (xs - cx) / fx
    y_cam = (ys - cy) / fy
    rays_cam = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=-1)
    points_cam = rays_cam * depths[:, None]
    points_cam_hom = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=-1)
    points_world = (pose @ points_cam_hom.T).T[:, :3]
    return points_world

def visualize_3d_points(points_world, title="3D Mask Points", save_path="backprojected_mask_3d.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_world[:, 0], points_world[:, 1], points_world[:, 2], c='red', s=0.5)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Saved 3D mask visualization to {save_path}")

def save_as_ply(points_world, ply_path="backprojected_mask_points.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"[INFO] Saved point cloud to {ply_path}")

# === MAIN ===
if __name__ == "__main__":
    transforms_json_path = "/home/thurapyaesone/Desktop/nerf2/data/Knee/transforms.json"
    mask_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/inferred_masks"
    depth_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/midas_depths"
    scale_factor = 10.0

    cameras, masks, filenames = load_training_cameras_and_masks(transforms_json_path, mask_folder)

    if len(masks) > 0:
        pose, intrinsics = cameras[0]
        mask = masks[0]
        base = os.path.splitext(filenames[0])[0]
        depth_path = os.path.join(depth_folder, f"{base}-midas_v21_small_256.pfm")
        if not os.path.exists(depth_path):
            print(f"[ERROR] Depth map missing: {depth_path}")
        else:
            depth = load_pfm(depth_path, normalize=True, scale_factor=scale_factor)
            points = backproject_mask_to_world_points(mask, depth, pose, intrinsics)
            if points.shape[0] == 0:
                print("[ERROR] No valid 3D points found in mask.")
            else:
                visualize_3d_points(points)
                save_as_ply(points)
    else:
        print("[ERROR] No valid masks found to visualize.")
