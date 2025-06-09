import json
import numpy as np
import cv2
import os
from tqdm import tqdm
import re

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
        mask_path = os.path.join(mask_folder, base_filename)

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            cameras.append((pose, intrinsics))
            masks.append(mask)
            filenames.append(base_filename)
        else:
            print(f"[Warning] Mask not found: {mask_path}")

    return cameras, masks, filenames

def load_rendered_cameras(bone_json_path):
    with open(bone_json_path, 'r') as f:
        bone = json.load(f)

    w = int(bone['render_width'])
    h = int(bone['render_height'])

    camera_poses = []
    for cam in bone['camera_path']:
        pose = np.array(cam['camera_to_world']).reshape(4, 4)
        fov = cam['fov']
        fx = fy = (w / 2) / np.tan(np.deg2rad(fov) / 2)
        cx = w / 2
        cy = h / 2
        camera_poses.append((pose, fx, fy, cx, cy, w, h))

    return camera_poses

def load_pfm(file_path, normalize=True, scale_factor=5.0):
    import re
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

    if normalize:
        # Prevent division by zero
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
        else:
            data[:] = 0  # fallback if depth is flat

        data *= scale_factor  # scale to match NeRF world units

    return data


def backproject_mask_to_points(mask, depth_map, pose, intrinsics):
    fx = intrinsics['fl_x']
    fy = intrinsics['fl_y']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    ys, xs = np.where(mask > 127)
    depths = depth_map[ys, xs]

    rays_cam = np.zeros((xs.shape[0], 3))
    rays_cam[:, 0] = (xs - cx) / fx
    rays_cam[:, 1] = (ys - cy) / fy
    rays_cam[:, 2] = 1.0

    points_cam = rays_cam * depths[:, None]
    points_cam_hom = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=-1)
    points_world = (pose @ points_cam_hom.T).T[:, :3]
    return points_world

def project_points_to_frame(points_world, render_pose, fx, fy, cx, cy, w, h):
    ones = np.ones((points_world.shape[0], 1))
    points_world_hom = np.concatenate([points_world, ones], axis=-1)

    w2c = np.linalg.inv(render_pose)
    points_cam = (w2c @ points_world_hom.T).T[:, :3]

    zs = points_cam[:, 2]
    valid = (zs > 1e-5) & np.isfinite(zs)
    points_cam = points_cam[valid]
    zs = zs[valid]

    if points_cam.shape[0] == 0:
        return np.zeros((h, w), dtype=np.uint8)

    xs = (fx * (points_cam[:, 0] / zs) + cx)
    ys = (fy * (points_cam[:, 1] / zs) + cy)

    valid_proj = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[valid_proj].astype(np.int32)
    ys = ys[valid_proj].astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    mask[ys[inside], xs[inside]] = 255

    return mask

def build_projected_masks(transforms_json_path, mask_folder, bone_json_path, output_mask_folder):
    os.makedirs(output_mask_folder, exist_ok=True)
    depth_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/midas_depths"
    render_rgb_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/rendered_frames"
    overlayed_output_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/overlayed_frames"
    os.makedirs(overlayed_output_folder, exist_ok=True)

    cameras, masks, filenames = load_training_cameras_and_masks(transforms_json_path, mask_folder)
    render_cameras = load_rendered_cameras(bone_json_path)

    all_points = []
    print("Backprojecting masks to 3D points...")
    for (pose, intrinsics), mask, name in tqdm(zip(cameras, masks, filenames), total=len(masks)):
        base = os.path.splitext(name)[0]
        depth_path = os.path.join(depth_folder, f"{base}-midas_v21_small_256.pfm")
        if not os.path.exists(depth_path):
            print(f"[Warning] Depth file not found: {depth_path}")
            continue
        depth = load_pfm(depth_path)
        points = backproject_mask_to_points(mask, depth, pose, intrinsics)
        all_points.append(points)

    all_points = np.concatenate(all_points, axis=0)

    print("Projecting 3D points to rendered frames and overlaying masks...")
    for idx, (render_pose, fx, fy, cx, cy, w, h) in tqdm(enumerate(render_cameras), total=len(render_cameras)):
        mask = project_points_to_frame(all_points, render_pose, fx, fy, cx, cy, w, h)
        render_path = os.path.join(render_rgb_folder, f"frame_{idx:04d}.png")
        overlay_path = os.path.join(overlayed_output_folder, f"frame_{idx:04d}.png")

        if os.path.exists(render_path):
            img = cv2.imread(render_path)
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(img, 1.0, mask_color, 0.5, 0)
            cv2.imwrite(overlay_path, blended)
        else:
            print(f"[Warning] Rendered frame not found: {render_path}")


if __name__ == "__main__":
    build_projected_masks(
        transforms_json_path="/home/thurapyaesone/Desktop/nerf2/data/Knee/transforms.json",
        mask_folder="/home/thurapyaesone/Desktop/nerf2/data/Knee/inferred_masks",
        bone_json_path="/home/thurapyaesone/Desktop/nerf2/data/Knee/camera_paths/Bone.json",
        output_mask_folder="/home/thurapyaesone/Desktop/nerf2/data/Knee/projected_masks"
    )
