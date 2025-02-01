import torch
import pytorch3d
import numpy as np
import matplotlib.pyplot as plt
import mcubes
from pytorch3d.renderer import TexturesVertex, PointLights, FoVPerspectiveCameras
from starter.utils import get_device, get_mesh_renderer
from q1_1 import create_render
import os

def create_torus_mesh(resolution: int = 256, grid_size: int = 64, minor_radius: float = 1.0, major_radius: float = 2.0, device: torch.device = None) -> pytorch3d.structures.Meshes:
    
    """Creates a torus mesh using implicit surface functions with iridescent coloring."""
    
    device = get_device() if device is None else device
    boundary = 5
    
    coordinates = torch.linspace(-boundary, boundary, grid_size)
    X, Y, Z = torch.meshgrid([coordinates] * 3)
    
    implicit_surface = (major_radius - torch.sqrt(X**2 + Y**2))**2 + Z**2 - minor_radius**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(implicit_surface), isovalue=0)
    
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)
    vertices = (vertices / grid_size) * (2 * boundary) - boundary
    
    vertex_colors = torch.nn.functional.normalize(vertices, dim=1)
    angles = torch.atan2(vertex_colors[:, 1], vertex_colors[:, 0])
    heights = vertex_colors[:, 2]
    
    r = 0.5 + 0.5 * torch.sin(angles * 2.0 + heights * 3.0)
    g = 0.5 + 0.5 * torch.sin(angles * 2.0 + heights * 3.0 + 2.0)
    b = 0.5 + 0.5 * torch.sin(angles * 2.0 + heights * 3.0 + 4.0)
    vertex_colors = torch.stack([r, g, b], dim=1)
    
    textures = TexturesVertex(vertex_colors.unsqueeze(0))
    return pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)


def create_fancy_mesh(resolution: int = 256, grid_size: int = 64, device: torch.device = None) -> pytorch3d.structures.Meshes:
    """Creates a hollowed cylinder with decorative patterns on the surface."""
    device = get_device() if device is None else device
    boundary = 4
    
    coordinates = torch.linspace(-boundary, boundary, grid_size)
    X, Y, Z = torch.meshgrid([coordinates] * 3)    
    outer_radius = 2.0
    inner_radius = 1.5
    height = 2.0
    pattern_freq = 8
    radius = torch.sqrt(X**2 + Y**2)
    
    pattern = 0.1 * torch.sin(pattern_freq * torch.atan2(Y, X)) * \
             torch.sin(2 * np.pi * Z / height)
    
    outer_cylinder = radius - (outer_radius + pattern)
    inner_cylinder = (inner_radius + pattern) - radius
    height_bounds = (torch.abs(Z) - height)
    
    surface = torch.maximum(
        torch.maximum(outer_cylinder, inner_cylinder),
        height_bounds
    )
    
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(surface), isovalue=0)
    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int64)
    vertices = (vertices / grid_size) * (2 * boundary) - boundary
    
    angle = torch.atan2(vertices[:, 1], vertices[:, 0])
    height_normalized = vertices[:, 2] / height
    radius_normalized = torch.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2) / outer_radius
    
    r = 0.5 + 0.5 * torch.sin(angle * pattern_freq + height_normalized * 5.0)
    g = 0.5 + 0.5 * torch.cos(height_normalized * 8.0 + radius_normalized * 3.0)
    b = 0.5 + 0.5 * torch.sin(radius_normalized * 4.0 - angle * 2.0)
    vertex_colors = torch.stack([r, g, b], dim=1)
    
    textures = TexturesVertex(vertex_colors.unsqueeze(0))
    return pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)



def render_implicit_mesh(mesh: pytorch3d.structures.Meshes, device: torch.device = None, resolution: int = 256, output_dir: str = "output", output_name: str = "result.gif", fps: int = 15, rotation_step: int = 5, camera_distance: float = 10, elevation: float = 0,) -> None:
    """Renders and saves a 360 degree animation of an implicit surface mesh."""
    device = get_device() if device is None else device
    mesh = mesh.to(device)    
    os.makedirs(output_dir, exist_ok=True)
    
    lights = PointLights(location=[[0, 0.0, -4.0]], device=device)
    renderer = get_mesh_renderer(image_size=resolution, device=device)
    
    initial_rotation, initial_translation = pytorch3d.renderer.look_at_view_transform(dist=camera_distance, elev=elevation, azim=180)
    cameras = FoVPerspectiveCameras(R=initial_rotation, T=initial_translation, device=device)
    frame = renderer(mesh, cameras=cameras, lights=lights)
    frame = frame[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    plt.imsave(os.path.join(output_dir, output_name.replace('.gif', '.jpg')), frame)
    create_render(input_mesh=mesh, render_size=resolution, output_dir=output_dir, output_name=output_name, rotation_step=rotation_step, animation_fps=fps, normalize_colors=True, camera_distance=camera_distance, camera_height=elevation)

def main():
    device = get_device()
    output_dir = "output"
    
    torus_mesh = create_torus_mesh(resolution=1024, device=device)
    render_implicit_mesh(torus_mesh, resolution=1024, output_dir=output_dir, output_name="q5_3_1.gif", camera_distance=6.0, elevation=20.0)
    
    cylinder_mesh = create_fancy_mesh(resolution=1024, device=device)
    render_implicit_mesh(cylinder_mesh, resolution=1024, output_dir=output_dir, output_name="q5_3_2.gif", camera_distance=8.0, elevation=30.0, rotation_step=3)

if __name__ == "__main__":
    main()