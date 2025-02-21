import argparse
import matplotlib.pyplot as plt
import pytorch3d
import torch
import os
import imageio
import numpy as np
from typing import Optional, List, Tuple, Union
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
import mcubes

# Constants
SAVE_PATH = "results/"
DEFAULT_IMAGE_SIZE = 512
DEFAULT_FPS = 15
DEFAULT_ANGLE_STEP = 5
DEFAULT_LIGHT_LOCATION = [[0, 0, -3]]
DEFAULT_BACKGROUND_COLOR = (1.0, 1.0, 1.0)
DEFAULT_MESH_COLOR = [1.0, 1.0, 1.0]
DEFAULT_LIGHT_INTENSITY = 0.7

def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_mesh_renderer(
    image_size: int = DEFAULT_IMAGE_SIZE,
    lights: Optional[PointLights] = None,
    device: Optional[torch.device] = None
) -> MeshRenderer:
    """Creates a Pytorch3D Mesh Renderer."""
    device = device or get_device()
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights)
    )

def load_mesh_from_obj(path: str = "data/cow_mesh.obj") -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads vertices and faces from an OBJ file."""
    vertices, faces, _ = load_obj(path)
    return vertices, faces.verts_idx

def create_colored_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    base_color: List[float],
    gradient_colors: Optional[Tuple[List[float], List[float]]] = None
) -> Meshes:
    """Creates a colored mesh from vertices and faces."""
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    textures = torch.ones_like(vertices)
    
    textures = textures * 0.3 + 0.3

    if gradient_colors:
        color1, color2 = gradient_colors
        z_coords = vertices[:, :, -1]
        z_min, z_max = z_coords.min().item(), z_coords.max().item()
        
        for i in range(len(z_coords[0])):
            alpha = (z_coords[0][i].item() - z_min) / (z_max - z_min)
            color = alpha * torch.tensor(color2) + (1 - alpha) * torch.tensor(color1)
            textures[0][i] = color * 0.3 + 0.3

    textures = textures * torch.tensor(base_color)
    return pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=TexturesVertex(textures)
    )

def render_360_degree_view(
    mesh: Optional[Meshes] = None,
    obj_path: str = "data/cow.obj",
    image_size: int = DEFAULT_IMAGE_SIZE,
    color: List[float] = DEFAULT_MESH_COLOR,
    device: Optional[torch.device] = None,
    save_path: str = SAVE_PATH,
    filename: str = "render_360.gif",
    angle_step: int = DEFAULT_ANGLE_STEP,
    fps: int = DEFAULT_FPS,
    gradient_colors: Optional[Tuple[List[float], List[float]]] = None,
    textured: bool = False,
    clip_values: bool = False,
    distance: float = 3,
    elevation: float = 0
) -> List[np.ndarray]:
    """Renders a 360-degree view of a 3D mesh."""
    device = device or get_device()
    renderer = create_mesh_renderer(image_size=image_size, device=device)

    if mesh is None and not textured:
        vertices, faces = load_mesh_from_obj(obj_path)
        mesh = create_colored_mesh(vertices, faces, color, gradient_colors)
    
    mesh = mesh.to(device)
    frames = []

    for angle in range(-180, 180, angle_step):
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=angle)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        lights = PointLights(
            location=DEFAULT_LIGHT_LOCATION,
            device=device,
            ambient_color=((DEFAULT_LIGHT_INTENSITY,) * 3,),
            diffuse_color=((1.0,) * 3,),
            specular_color=((1.0,) * 3,)
        )

        render = renderer(mesh, cameras=cameras, lights=lights)
        frame = render.cpu().detach().numpy()[0, ..., :3]

        if clip_values:
            frame = frame.clip(0, 1)
        
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimsave(os.path.join(save_path, filename), frames, fps=fps, loop=0)
    return frames

def create_points_renderer(
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: Optional[torch.device] = None,
    radius: float = 0.01,
    background_color: Tuple[float, float, float] = DEFAULT_BACKGROUND_COLOR
) -> PointsRenderer:
    """Creates a renderer for point clouds."""
    device = device or get_device()
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius
    )
    return PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color)
    )

def visualize_voxels(
    voxels: torch.Tensor,
    filename: str = "voxel_viz.gif",
    image_size: int = DEFAULT_IMAGE_SIZE,
    distance: float = 3,
    elevation: float = 0
) -> None:
    """Visualizes voxel data as a mesh."""
    device = get_device()
    mesh = pytorch3d.ops.cubify(voxels, thresh=0.5, device=device)

    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    
    vertex_colors = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    vertex_colors = vertex_colors * 0.3 + 0.3
    textures = pytorch3d.renderer.TexturesVertex(vertex_colors.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)
    lights = PointLights(
        location=[[0, 0.0, -4.0]],
        device=device,
        ambient_color=((DEFAULT_LIGHT_INTENSITY,) * 3,),
        diffuse_color=((1.0,) * 3,),
        specular_color=((1.0,) * 3,)
    )
   
    renderer = create_mesh_renderer(image_size=image_size, device=device)
    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=180)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    render = renderer(mesh, cameras=cameras, lights=lights)
    render = render[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    
    plt.style.use('dark_background')
    plt.imsave(os.path.join(SAVE_PATH, "voxel_render.jpg"), render)

    render_360_degree_view(
        mesh=mesh,
        filename=filename,
        device=device,
        image_size=image_size,
        distance=distance,
        elevation=elevation,
        clip_values=True,
        color=DEFAULT_MESH_COLOR
    )

def visualize_point_cloud(
    point_cloud: Pointclouds,
    image_size: int = DEFAULT_IMAGE_SIZE,
    background_color: Tuple[float, float, float] = DEFAULT_BACKGROUND_COLOR,
    save_path: str = SAVE_PATH,
    filename: str = "point_cloud.gif",
    device: Optional[torch.device] = None,
    fps: int = DEFAULT_FPS,
    angle_step: int = DEFAULT_ANGLE_STEP,
    distance: float = 7,
    elevation: float = 0,
    upside_down: bool = False
) -> None:
    """Visualizes a point cloud."""
    device = device or get_device()
    renderer = create_points_renderer(
        image_size=image_size,
        background_color=background_color,
        device=device
    )

    angle = torch.Tensor([0, 0, np.pi if upside_down else 0])
    rotation = pytorch3d.transforms.euler_angles_to_matrix(angle, "XYZ")
    frames = []

    for angle in range(-180, 180, angle_step):
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=angle)
        cameras = FoVPerspectiveCameras(R=R @ rotation, T=T, device=device)
        lights = PointLights(
            location=DEFAULT_LIGHT_LOCATION,
            device=device,
            ambient_color=((DEFAULT_LIGHT_INTENSITY,) * 3,),
            diffuse_color=((1.0,) * 3,),
            specular_color=((1.0,) * 3,)
        )

        render = renderer(point_cloud, cameras=cameras, lights=lights)
        frame = render.cpu().numpy()[0, ..., :3]
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)

    imageio.mimsave(os.path.join(save_path, filename), frames, fps=fps)

def visualize_point_cloud_from_points(
    points: torch.Tensor,
    filename: str = "points_viz.gif",
    image_size: int = DEFAULT_IMAGE_SIZE,
    distance: float = 1.5,
    elevation: float = 0
) -> None:
    """Creates and visualizes a point cloud from points."""
    device = get_device()
    points = points.detach()[0]
    
    colors = (points - points.min()) / (points.max() - points.min())
    colors = colors * 0.3 + 0.3

    point_cloud = Pointclouds(
        points=[points],
        features=[colors]
    ).to(device)

    visualize_point_cloud(
        point_cloud,
        image_size=image_size,
        filename=filename,
        device=device,
        distance=distance,
        elevation=elevation,
        background_color=DEFAULT_BACKGROUND_COLOR
    )

def visualize_mesh_model(
    mesh: Meshes,
    filename: str = "mesh_viz.gif",
    image_size: int = DEFAULT_IMAGE_SIZE,
    distance: float = 1,
    elevation: float = 0
) -> None:
    """Visualizes a mesh model."""
    device = get_device()
    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    
    vertex_colors = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    vertex_colors = vertex_colors * 0.3 + 0.3
    textures = pytorch3d.renderer.TexturesVertex(vertex_colors.unsqueeze(0))

    processed_mesh = pytorch3d.structures.Meshes(
        [vertices],
        [faces],
        textures=textures
    ).to(device)

    render_360_degree_view(
        mesh=processed_mesh,
        filename=filename,
        device=device,
        image_size=image_size,
        distance=distance,
        elevation=elevation,
        clip_values=True,
        color=DEFAULT_MESH_COLOR
    )

if __name__ == "__main__":
    pass