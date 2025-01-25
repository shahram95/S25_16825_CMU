import os
import torch
import pytorch3d
import numpy as np
import imageio
from tqdm import tqdm
from typing import List, Optional,Tuple
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    TexturesVertex
)
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

def generate_vertex_gradient(mesh_vertices: torch.Tensor, start_color: List[float], end_color: List[float]) -> torch.Tensor:
    '''
    Create vertex colors that gradient between two colors based on z-height.
    '''
    vertex_heights = mesh_vertices[:, :, -1]
    min_height, max_height = vertex_heights.min().item(), vertex_heights.max().item()
    normalized_heights = (vertex_heights[0] - min_height) / (max_height - min_height)

    start_color_tensor = torch.tensor(start_color)
    end_color_tensor = torch.tensor(end_color)
    vertex_colors = torch.ones_like(mesh_vertices)

    for idx,height in enumerate(normalized_heights):
        blend_factor = height.item()
        vertex_color = blend_factor * end_color_tensor + (1 - blend_factor) * start_color_tensor
        vertex_colors[0][idx] = vertex_color
    return vertex_color

def create_render(model_path: str='data/cow.obj', input_mesh: Optional[Meshes]=None, render_size: int=256, mesh_color: List[float]=[0.7,0.7,1], device: Optional[torch.device]=None, output_dir: str='output/', output_name: str = 'render.gif', rotation_step: int=5, animation_fps: int=15, gradient_colors: Optional[Tuple[List[float], List[float]]]=None, has_texture: bool=False, normalize_colors: bool=False, camera_distance: float=3.0, camera_height: float=0.0) -> List[np.ndarray]:
    '''
    Create a 360-degree animation of a 3D model
    '''
    device = get_device() if device is None else device
    renderer = get_mesh_renderer(image_size=render_size, device=device)

    if input_mesh is None and not has_texture:
        vertices, faces = load_cow_mesh(model_path)
        vertices = vertices.unsqueeze(0)
        faces = faces.unsqueeze(0)
        vertex_colors = torch.ones_like(vertices)

        if gradient_colors is not None:
            vertex_colors = generate_vertex_gradient(vertices, gradient_colors[0], gradient_colors[1])
        
        vertex_colors = vertex_colors * torch.tensor(mesh_color)
        input_mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=TexturesVertex(vertex_colors))
    
    input_mesh = input_mesh.to(device)
    os.makedirs(output_dir, exist_ok=True)
    frames = []

    for angle in tqdm(range(-180, 180, rotation_step), desc='Generating frames'):
        rotation, translation = look_at_view_transform(dist=camera_distance, elev=camera_height, azim=angle)
        cameras = FoVPerspectiveCameras(R=rotation, T=translation, device=device)
        lights = PointLights(location=[[0,0,-3]], device=device)

        # Get the render output
        frame = renderer(input_mesh, cameras=cameras, lights=lights)
        
        # Convert to numpy and extract RGB channels (remove alpha)
        frame = frame.cpu().numpy()[0, ..., :3]  # Shape now: (H, W, 3)

        if normalize_colors:
            frame = np.clip(frame, 0, 1)
        
        # Convert to uint8 for saving
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)
    
    output_path = os.path.join(output_dir, output_name)
    imageio.mimsave(output_path, frames, fps=animation_fps, loop=0)
    return frames

if __name__ == "__main__":
    create_render(render_size=1024, output_dir="output", output_name="q1_1.gif")