import torch
from typing import List, Optional, Tuple
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from q1_1 import create_render

def create_3d_shape(shape_type: str = "tetrahedron", base_color: List[float] = [0.0, 0.4, 0.4], scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create vertices and faces for basic 3D shapes.
    """
    if shape_type.lower() == "tetrahedron":
        vertices = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 2.0/3**0.5, -1.0/3],
            [-1.0, -1.0/3**0.5, -1.0/3],
            [1.0, -1.0/3**0.5, -1.0/3]
        ]) * scale

        faces = torch.tensor([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 2, 3]
        ])
        
    elif shape_type.lower() == "cube":
        vertices = torch.tensor([
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0]
        ]) * scale * 0.5

        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5]
        ])
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")
    
    return vertices, faces

def visualize_3d_shape(shape_type: str, output_name: str, render_size: int = 256, base_color: List[float] = [0.0, 0.4, 0.4], gradient_colors: Optional[Tuple[List[float], List[float]]] = None, camera_distance: float = 3.0, camera_height: float = 30.0, scale: float = 1.0) -> None:
    """
    Create and render a 3D geometric shape using the create_render function.    
    """
    vertices, faces = create_3d_shape(shape_type, base_color, scale)
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    
    create_render(input_mesh=Meshes(verts=vertices, faces=faces, textures=TexturesVertex(torch.ones_like(vertices) * torch.tensor(base_color))), render_size=render_size, output_name=output_name, gradient_colors=gradient_colors, camera_distance=camera_distance, camera_height=camera_height)

if __name__ == "__main__":
    visualize_3d_shape(shape_type="tetrahedron", output_name="q2_1.gif", render_size=1024, base_color=[0.0, 0.5, 0.8], camera_height=30.0, gradient_colors=([0.0, 0.3, 0.6], [0.0, 0.7, 1.0]))
    visualize_3d_shape(shape_type="cube", output_name="q2_2.gif", render_size=1024, base_color=[0.8, 0.2, 0.2], camera_height=20.0, gradient_colors=([0.6, 0.1, 0.1], [1.0, 0.3, 0.3]))