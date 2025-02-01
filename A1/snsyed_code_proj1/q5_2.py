import torch
import pytorch3d
import numpy as np
import imageio
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import FoVPerspectiveCameras, PointLights
from starter.utils import get_device, get_points_renderer
from starter.render_generic import load_rgbd_data

def create_torus_point_cloud(num_samples: int = 200, inner_radius: float = 1.0, outer_radius: float = 2.0, device: torch.device = None) -> Pointclouds:
    """Creates a torus with rainbow-like coloring based on angular position."""
    device = get_device() if device is None else device
    
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    phi_grid, theta_grid = torch.meshgrid(phi, theta)
    
    x = inner_radius * (torch.cos(theta_grid) + outer_radius) * torch.cos(phi_grid)
    y = inner_radius * (torch.cos(theta_grid) + outer_radius) * torch.sin(phi_grid)
    z = inner_radius * torch.sin(theta_grid)
    
    point_positions = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    
    colors_r = 0.5 + 0.5 * torch.cos(phi_grid.flatten())
    colors_g = 0.5 + 0.5 * torch.sin(theta_grid.flatten())
    colors_b = 0.5 + 0.5 * torch.cos(phi_grid.flatten() + theta_grid.flatten())
    point_colors = torch.stack((colors_r, colors_g, colors_b), dim=1)
    
    return Pointclouds(points=[point_positions], features=[point_colors]).to(device)

def create_saddle_point_cloud(num_samples: int = 200, scale: float = 2.0, device: torch.device = None) -> Pointclouds:
    """Creates a hyperbolic paraboloid (saddle shape) with gradient coloring."""
    device = get_device() if device is None else device
    
    x = torch.linspace(-scale, scale, num_samples)
    y = torch.linspace(-scale, scale, num_samples)
    x_grid, y_grid = torch.meshgrid(x, y)
    
    z = (x_grid * x_grid - y_grid * y_grid) / scale
    
    point_positions = torch.stack((x_grid.flatten(), y_grid.flatten(), z.flatten()), dim=1)
    
    radius = torch.sqrt(x_grid.flatten()**2 + y_grid.flatten()**2)
    height = z.flatten()
    
    colors_r = torch.sigmoid(radius / scale)
    colors_g = torch.sigmoid(-height / scale)
    colors_b = torch.sigmoid(height / scale)
    point_colors = torch.stack((colors_r, colors_g, colors_b), dim=1)
    
    return Pointclouds(points=[point_positions], features=[point_colors]).to(device)


def render_parametric_shape(point_cloud: Pointclouds, render_size: int = 256, background_color: tuple = (1, 1, 1), output_dir: str = "output/", output_name: str = "shape.gif", device: torch.device = None, fps: int = 15, rotation_step: int = 5, camera_distance: float = 8.0, camera_elevation: float = 0.0) -> None:
    """Creates a 360-degree animation of a parametric shape point cloud."""
    device = get_device() if device is None else device
    point_cloud = point_cloud.to(device)
    renderer = get_points_renderer(image_size=render_size, background_color=background_color).to(device)
    frames = []

    for angle in range(-180, 180, rotation_step):
        camera_rotation, camera_translation = pytorch3d.renderer.look_at_view_transform(dist=camera_distance, elev=camera_elevation, azim=angle, device=device)
        camera = FoVPerspectiveCameras(R=camera_rotation, T=camera_translation, device=device)
        lights = PointLights(location=[[0, 0, -3]], device=device)
        
        frame = renderer(point_cloud, cameras=camera, lights=lights)
        frame = (frame[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)
    
    imageio.mimsave(output_dir + output_name, frames, fps=fps)

def main():
    device = get_device()
    
    torus_cloud = create_torus_point_cloud(num_samples=800, device=device)
    render_parametric_shape(torus_cloud, render_size=1024, output_name="q5_2_1.gif", background_color=(0.95, 0.95, 1.0), camera_distance=6.0, camera_elevation=20.0, device=device)
    
    saddle_cloud = create_saddle_point_cloud(num_samples=600, device=device)
    render_parametric_shape(saddle_cloud, render_size=1024, output_name="q5_2_2.gif", background_color=(0.95, 0.95, 1.0), camera_distance=8.0, camera_elevation=30.0, rotation_step=3, device=device)

if __name__ == "__main__":
    main()