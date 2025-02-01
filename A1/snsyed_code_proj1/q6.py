import os
import torch
import pytorch3d
import numpy as np
import matplotlib.pyplot as plt
import imageio
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    TexturesUV,
    TexturesVertex,
    look_at_view_transform,
    Materials
)
from starter.utils import get_device, get_mesh_renderer

def load_and_process_texture(texture_path: str) -> np.ndarray:
    """Loads and processes a texture image for rendering."""
    texture = plt.imread(texture_path)
    
    if texture.dtype == np.uint8:
        texture = texture.astype(np.float32) / 255.0
    
    if len(texture.shape) == 2:
        texture = np.stack([texture] * 3, axis=-1)
    elif texture.shape[-1] == 4:
        texture = texture[..., :3]
    
    return texture

def create_cottage_scene(model_path: str = "data/cottage_obj.obj", diffuse_map: str = "data/cottage_diffuse.png", normal_map: str = "data/cottage_normal.png", image_size: int = 512, output_name: str = "cottage_scene.gif", device: Optional[torch.device] = None, output_dir: str = "output", animation_fps: int = 15, background_color: List[float] = [0.1, 0.1, 0.2]) -> None:
    """Creates interior and exterior views of the cottage."""
    device = get_device() if device is None else device
    os.makedirs(output_dir, exist_ok=True)

    vertices, faces, aux = load_obj(model_path, device=device, load_textures=True, create_texture_atlas=True, texture_atlas_size=4, texture_wrap="repeat")

    diffuse_texture = load_and_process_texture(diffuse_map)
    diffuse_texture = torch.from_numpy(diffuse_texture)[None].to(device)
    
    textures = TexturesUV(maps=diffuse_texture, faces_uvs=faces.textures_idx.unsqueeze(0), verts_uvs=aux.verts_uvs.unsqueeze(0),).to(device)

    cottage_mesh = Meshes(verts=vertices.unsqueeze(0), faces=faces.verts_idx.unsqueeze(0), textures=textures).to(device)

    frames = []
    renderer = get_mesh_renderer(image_size=image_size)

    def render_with_background(mesh, cameras, lights):
        """Helper function to render with consistent background."""
        rend = renderer(mesh, cameras=cameras, lights=lights)
        frame = rend[0, ..., :3].cpu().numpy()
        mask = (frame.sum(axis=-1) < 1e-4)[..., None]
        frame = frame * (1 - mask) + np.array(background_color) * mask
        return (frame * 255).astype(np.uint8)

    interior_height = -2.0
    interior_distance = 2.0
    
    for angle in tqdm(range(0, 360, 5), desc="Interior view"):
        R, T = look_at_view_transform(dist=interior_distance, elev=15, azim=angle, device=device)
        
        camera_T = T.clone()
        camera_T[..., 1] += interior_height
        
        cameras = FoVPerspectiveCameras(R=R, T=camera_T, device=device)
        
        lights = PointLights(location=torch.tensor([[0, interior_height + 0.5, 0]], device=device), diffuse_color=torch.tensor([[1.0, 0.9, 0.8]], device=device), device=device)
        frames.append(render_with_background(cottage_mesh, cameras, lights))
        
    start_dist = interior_distance
    end_dist = 30.0
    num_transition_frames = 90
    
    for i in tqdm(range(num_transition_frames), desc="Zooming out"):
        progress = i / (num_transition_frames - 1)
        current_dist = start_dist + (end_dist - start_dist) * progress
        current_height = interior_height + progress * 3
        current_elev = 15 + progress * 25
        
        R, T = look_at_view_transform(dist=current_dist, elev=current_elev, azim=0, device=device)
        
        camera_T = T.clone()
        camera_T[..., 1] += current_height
        
        cameras = FoVPerspectiveCameras(R=R, T=camera_T, device=device)
        
        light_pos = torch.tensor([[0, current_height + 1.0, -current_dist/2]], device=device)
        light_color = torch.tensor([[1.0, 0.95 + 0.05 * progress, 0.9 + 0.1 * progress]], device=device)
        lights = PointLights(location=light_pos, diffuse_color=light_color, device=device)
        frames.append(render_with_background(cottage_mesh, cameras, lights))

    for angle in tqdm(range(0, 360, 5), desc="Exterior view"):
        R, T = look_at_view_transform(dist=30.0, elev=40, azim=angle, device=device)
        
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        
        lights = PointLights(location=torch.tensor([[0, 5.0, -5.0]], device=device), diffuse_color=torch.tensor([[1.0, 1.0, 1.0]], device=device), device=device)
        frames.append(render_with_background(cottage_mesh, cameras, lights))

    output_path = os.path.join(output_dir, output_name)
    imageio.mimsave(output_path, frames, fps=animation_fps)

def main():
    create_cottage_scene(image_size=1024, diffuse_map="data/cottage_diffuse.png", normal_map="data/cottage_normal.png", output_name="cottage_scene.gif", animation_fps=20,background_color=[0.1, 0.1, 0.2])

if __name__ == "__main__":
    main()