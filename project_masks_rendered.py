import json
import numpy as np
import cv2
import os
from tqdm import tqdm
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    print(f"[INFO] Loaded {len(masks)} masks and poses")
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

    if subsample_rate > 1:
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

def project_to_rendered_frame(points_world, render_pose, fx, fy, cx, cy, w, h):
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

    xs = (fx * (points_cam[:, 0] / zs) + cx).astype(np.int32)
    ys = (fy * (points_cam[:, 1] / zs) + cy).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    mask[ys[inside], xs[inside]] = 255

    return mask

def visualize_3d_points(points_world, title="3D Mask Points"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    xs = points_world[:, 0]
    ys = points_world[:, 1]
    zs = points_world[:, 2]
    ax.scatter(xs, ys, zs, c='red', s=0.5)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig("backprojected_mask_3d.png")
    print("[INFO] Saved 3D mask visualization to backprojected_mask_3d.png")

if __name__ == "__main__":
    transforms_json_path = "/home/thurapyaesone/Desktop/nerf2/data/Knee/transforms.json"
    mask_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/inferred_masks"
    depth_folder = "/home/thurapyaesone/Desktop/nerf2/data/Knee/midas_depths"
    bone_json_path = "/home/thurapyaesone/Desktop/nerf2/data/Knee/camera_paths/Bone.json"
    rendered_output_path = "/home/thurapyaesone/Desktop/nerf2/data/Knee/overlayed_frames"
    rendered_frames_path = "/home/thurapyaesone/Desktop/nerf2/data/Knee/rendered_frames"
    output_video_path = "/home/thurapyaesone/Desktop/nerf2/data/Knee/overlayed_render.mp4"
    os.makedirs(rendered_output_path, exist_ok=True)

    scale_factor = 10.0
    cameras, masks, filenames = load_training_cameras_and_masks(transforms_json_path, mask_folder)
    render_cameras = load_rendered_cameras(bone_json_path)

    if len(masks) > 0:
        pose, intrinsics = cameras[0]
        mask = masks[0]
        base = os.path.splitext(filenames[0])[0]
        depth_path = os.path.join(depth_folder, f"{base}-midas_v21_small_256.pfm")
        depth = load_pfm(depth_path, normalize=True, scale_factor=scale_factor)
        points = backproject_mask_to_world_points(mask, depth, pose, intrinsics)
        visualize_3d_points(points)

        print("[INFO] Projecting and overlaying on rendered frames...")
        for idx, (render_pose, fx, fy, cx, cy, w, h) in tqdm(enumerate(render_cameras), total=len(render_cameras)):
            mask_proj = project_to_rendered_frame(points, render_pose, fx, fy, cx, cy, w, h)
            render_path = os.path.join(rendered_frames_path, f"frame_{idx:04d}.png")
            overlay_path = os.path.join(rendered_output_path, f"frame_{idx:04d}.png")

            if os.path.exists(render_path):
                image = cv2.imread(render_path)
                mask_color = cv2.applyColorMap(mask_proj, cv2.COLORMAP_JET)
                blended = cv2.addWeighted(image, 1.0, mask_color, 0.5, 0)
                cv2.imwrite(overlay_path, blended)
            else:
                print(f"[Warning] Missing rendered frame: {render_path}")

        # Step 4: Generate video from overlayed frames
        print("[INFO] Generating video...")
        frame_files = sorted([f for f in os.listdir(rendered_output_path) if f.endswith('.png')])
        if frame_files:
            first_frame = cv2.imread(os.path.join(rendered_output_path, frame_files[0]))
            height, width, _ = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
            for fname in frame_files:
                frame = cv2.imread(os.path.join(rendered_output_path, fname))
                video.write(frame)
            video.release()
            print(f"[INFO] Saved overlay video to: {output_video_path}")
        else:
            print("[ERROR] No frames found to write video.")
    else:
        print("[ERROR] No valid masks found to project.")
