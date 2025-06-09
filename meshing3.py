import trimesh
import plotly.graph_objects as go

# === Path to your PLY point cloud ===
full_pcd_path = "/home/thurapyaesone/Desktop/nerf2/nerfstudio-method-semanticnerf/exports/pcd/point_cloud.ply"

# === Load point cloud using trimesh
cloud = trimesh.load(full_pcd_path, process=False)

# === Get point coordinates
points = cloud.vertices  # Nx3 array

# === Optional: Check if colors exist
if hasattr(cloud, 'colors') and cloud.colors is not None:
    colors = cloud.colors / 255.0  # Normalize RGB
else:
    colors = points[:, 2]  # Color by Z value if no RGB

# === Create scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(
        size=1.5,  # Increase point size for better visibility
        color=colors,
        colorscale='Viridis' if colors.ndim == 1 else None,
        opacity=0.9
    )
)])

# === Compute bounds for tight framing
x_range = [points[:, 0].min(), points[:, 0].max()]
y_range = [points[:, 1].min(), points[:, 1].max()]
z_range = [points[:, 2].min(), points[:, 2].max()]
center = [(x_range[0] + x_range[1]) / 2,
          (y_range[0] + y_range[1]) / 2,
          (z_range[0] + z_range[1]) / 2]

# Set tighter scene and camera for better zoom
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range),
        zaxis=dict(visible=False, range=z_range),
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),  # Move camera closer to the object
            center=dict(x=0, y=0, z=0)
        )
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    title="Interactive 3D Point Cloud with Better Zoom"
)

# === Show interactive plot
fig.show()
