import torch
import pytorch3d
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights
)
from starter.utils import get_device, get_mesh_renderer
import os

def create_camera_transform(model_path: str = "data/cow_with_axis.obj", render_size: int = 256, rotation_matrix: Union[List[List[float]], torch.Tensor] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation_vector: Union[List[float], torch.Tensor] = [0, 0, 0], device: Optional[torch.device] = None, output_dir: str = "output", output_name: str = "camera_transform.jpg") -> np.ndarray:
    """
    Render a 3D model with specified camera transformations using perspective projection.
    """
    device = get_device() if device is None else device
    mesh = pytorch3d.io.load_objs_as_meshes([model_path]).to(device)
    rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
    translation_vector = torch.tensor(translation_vector, dtype=torch.float32)

    base_rotation = torch.eye(3)
    final_rotation = rotation_matrix @ base_rotation
    base_camera_position = torch.tensor([0.0, 0.0, 3.0])
    final_translation = rotation_matrix @ base_camera_position + translation_vector

    renderer = get_mesh_renderer(image_size=render_size)
    
    camera = FoVPerspectiveCameras(R=final_rotation.t().unsqueeze(0), T=final_translation.unsqueeze(0), device=device)

    light_position = torch.tensor([[0.0, 0.0, -3.0]], device=device)
    lights = PointLights(location=light_position, device=device)
    rendered_image = renderer(mesh, cameras=camera, lights=lights)    
    rendered_image = rendered_image[0, ..., :3].cpu().numpy()


    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    plt.imsave(output_path, rendered_image)

    return rendered_image


if __name__ == "__main__":
    angle1 = torch.tensor([0, 0, np.pi/2])
    R1 = pytorch3d.transforms.euler_angles_to_matrix(angle1, "XYZ")
    create_camera_transform(
        rotation_matrix=R1,
        output_name="q4_1.jpg"
    )

    create_camera_transform(
        translation_vector=[0, 0, 2],
        output_name="q4_2.jpg"
    )

    create_camera_transform(
        translation_vector=[0.4, -0.45, -0.03],
        output_name="q4_3.jpg"
    )

    angle4 = torch.tensor([0, -np.pi/2, 0])
    R4 = pytorch3d.transforms.euler_angles_to_matrix(angle4, "XYZ")
    create_camera_transform(
        rotation_matrix=R4,
        translation_vector=[4, 0, 4],
        output_name="q4_4.jpg"
    )