import torch
import pytorch3d
import numpy as np
import os
import imageio
from typing import Optional, List
from tqdm import tqdm
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    look_at_view_transform,
    TexturesVertex
)
from starter.utils import get_device, get_points_renderer, get_mesh_renderer

def create_point_samples(
    model_path: str = "data/cow.obj",
    num_samples: int = 400,
    device: Optional[torch.device] = None
) -> Pointclouds:
    """Creates point cloud by sampling points from mesh faces using area-weighted probability."""
    device = get_device() if device is None else device
    
    vertices, faces, _ = load_obj(
        model_path,
        device=device,
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=4
    )
    
    face_vertices = vertices[faces.verts_idx]
    edge1 = face_vertices[:, 1] - face_vertices[:, 0]
    edge2 = face_vertices[:, 2] - face_vertices[:, 0]
    areas = 0.5 * torch.norm(torch.cross(edge1, edge2, dim=1), dim=1)
    
    face_probs = areas / areas.sum()
    selected_faces = face_vertices[torch.multinomial(face_probs, num_samples, replacement=True)]
    
    u = torch.sqrt(torch.rand(num_samples, device=device))
    v = torch.rand(num_samples, device=device)
    
    sampled_points = (
        (1 - u).unsqueeze(1) * selected_faces[:, 0] +
        (u * (1 - v)).unsqueeze(1) * selected_faces[:, 1] +
        (u * v).unsqueeze(1) * selected_faces[:, 2]
    )
    
    point_colors = torch.zeros_like(sampled_points)
    normalized_points = (sampled_points - sampled_points.min()) / (sampled_points.max() - sampled_points.min())
    point_colors[..., 0] = 0.7 * normalized_points[..., 2]
    point_colors[..., 1] = 0.5 * normalized_points[..., 0]
    point_colors[..., 2] = 0.8 * normalized_points[..., 1]
    
    return Pointclouds(
        points=sampled_points.unsqueeze(0),
        features=point_colors.unsqueeze(0)
    )

def render_points(
    point_cloud: Pointclouds,
    output_path: str,
    image_size: int = 256,
    background_color: List[float] = [0.0, 0.0, 0.0],
    device: Optional[torch.device] = None,
    fps: int = 15,
    rotation_step: int = 5,
    camera_distance: float = 10.0,
    camera_height: float = 10.0
) -> List[np.ndarray]:
    """Renders rotating animation of point cloud."""
    device = get_device() if device is None else device
    renderer = get_points_renderer(
        image_size=image_size,
        background_color=background_color
    ).to(device)
    
    frames = []
    for angle in tqdm(range(-180, 180, rotation_step), desc="Rendering frames"):
        R, T = look_at_view_transform(
            dist=camera_distance,
            elev=camera_height,
            azim=angle
        )
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        lights = PointLights(
            location=[[0, camera_height, -3]],
            diffuse_color=[[1.0, 0.9, 0.8]],
            device=device
        )
        
        rend = renderer(point_cloud.to(device), cameras=cameras, lights=lights)
        frame = rend[0, ..., :3].cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    return frames

def render_mesh(
    output_path: str,
    image_size: int = 256,
    device: Optional[torch.device] = None
) -> List[np.ndarray]:
    """Renders original mesh animation."""
    device = get_device() if device is None else device
    
    vertices, faces = load_obj("data/cow.obj")[0:2]
    vertices = vertices.unsqueeze(0)
    faces = faces.verts_idx.unsqueeze(0)
    
    textures = torch.ones_like(vertices)
    textures = textures * torch.tensor([0.7, 0.7, 1.0])
    
    mesh = Meshes(
        verts=vertices,
        faces=faces,
        textures=TexturesVertex(textures)
    ).to(device)
    
    renderer = get_mesh_renderer(image_size=image_size)
    frames = []
    
    for angle in tqdm(range(-180, 180, 5), desc="Rendering mesh"):
        R, T = look_at_view_transform(dist=10.0, elev=10.0, azim=angle)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        lights = PointLights(location=[[0, 10.0, -3]], device=device)
        
        rend = renderer(mesh, cameras=cameras, lights=lights)
        frame = rend[0, ..., :3].cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)
    
    imageio.mimsave(output_path, frames, fps=15)
    return frames

def create_comparison(
    output_dir: str = "output",
    sample_sizes: List[int] = [10, 100, 1000, 10000],
    image_size: int = 1024,
    device: Optional[torch.device] = None
) -> None:
    """Creates full comparison visualization of different sample sizes and original mesh."""
    device = get_device() if device is None else device
    os.makedirs(output_dir, exist_ok=True)
    
    all_frames = []
    
    # Render point clouds
    for size in sample_sizes:
        point_cloud = create_point_samples(num_samples=size, device=device)
        frames = render_points(
            point_cloud=point_cloud,
            output_path=f"{output_dir}/q7_{size}.gif",
            image_size=image_size,
            device=device
        )
        all_frames.append(frames)
    
    # Render original mesh
    mesh_frames = render_mesh(
        output_path=f"{output_dir}/q7_original.gif",
        image_size=image_size,
        device=device
    )
    all_frames.append(mesh_frames)
    
    # Create combined visualization
    frame_count = min(len(frames) for frames in all_frames)
    combined_frames = []
    
    for i in range(frame_count):
        row = [frames[i] for frames in all_frames]
        combined = np.hstack(row)
        combined_frames.append(combined)
    
    imageio.mimsave(f"{output_dir}/q7_comparison.gif", combined_frames, fps=15)

if __name__ == "__main__":
    create_comparison(
        output_dir="output",
        sample_sizes=[10, 100, 1000, 10000],
        image_size=1024
    )