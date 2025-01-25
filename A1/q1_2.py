import torch
import numpy as np
import imageio
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import List, Optional, Union
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights
)
from pytorch3d.io import load_objs_as_meshes
from starter.utils import get_device, get_mesh_renderer

def create_dolly_zoom_effect(model_path: str="data/cow_on_plane.obj", render_size: int=256, frame_count: int=10, animation_duration: float=3.0, device: Optional[torch.device]=None, output_path: str='output/dolly.gif', input_mesh: Optional[Meshes]=None, min_fov: float=5.0, max_fov: float=120.0) -> List[np.ndarray]:
    '''
    Create a dolly zoom effect animation, where the camera's field of view and distance are adjusted to keep the subject size constant while changing perspective.
    '''
    device = get_device() if device is None else device
    mesh = input_mesh if input_mesh is not None else load_objs_as_meshes([model_path])
    mesh = mesh.to(device)

    renderer = get_mesh_renderer(image_size=render_size, device=device)
    light_position = torch.tensor([[0.0, 0.0, -3.0]], device=device)
    lights = PointLights(location=light_position, device=device)

    fov_sequence = torch.linspace(min_fov, max_fov, frame_count)
    rendered_frames = []

    for current_fov in tqdm(fov_sequence, desc="Rendering dolly zoom"):
        camera_distance = 5.0 / (2.0 * np.tan(np.radians(current_fov.item()) / 2.0))
        camera_position = torch.tensor([[0, 0, camera_distance]], device=device)
        
        camera = FoVPerspectiveCameras(
            fov=current_fov,
            T=camera_position,
            device=device
        )
        
        render_output = renderer(mesh, cameras=camera, lights=lights)
        frame = render_output[0, ..., :3].cpu().numpy()
        rendered_frames.append(frame)
    
    annotated_frames = []
    for idx, frame in enumerate(rendered_frames):
        image = Image.fromarray((frame * 255).astype(np.uint8))
        drawer = ImageDraw.Draw(image)
        text_position = (20,20)
        text_color = (255,0,0)
        # drawer.text(text_position, f'FOV: {fov_sequence[idx]:.2f}°', fill=text_color)
        annotated_frames.append(np.array(image))
    
    fps = frame_count / animation_duration
    imageio.mimsave(output_path, annotated_frames, fps=fps)
    return annotated_frames

if __name__ == '__main__':
    create_dolly_zoom_effect(render_size=1024, frame_count=30, animation_duration=3, output_path="output/q1_2.gif")