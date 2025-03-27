import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
import numpy as np

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_seg (verts, labels, path, device):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1,10000,3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15)

def viz_cls(verts, label=None, path="output.gif", device="cpu"):
    """
    Visualize classification result
    output: a 360-degree gif
    
    Args:
        verts: tensor of shape (N, 3) containing 3D point coordinates
        label: class label (int) or None
        path: output path for the gif
        device: device to use for rendering
    """
    
    
    image_size = 256
    background_color = (1, 1, 1)
    
    # Class colors: red for chair, green for vase, blue for lamp
    class_colors = [
        [1.0, 0.0, 0.0],  # Chair (Red)
        [0.0, 1.0, 0.0],  # Vase (Green)
        [0.0, 0.0, 1.0]   # Lamp (Blue)
    ]
    
    # Construct various camera viewpoints
    dist = 3
    elev = 30  # Slightly elevated view for better visualization
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=dist, elev=elev, azim=azim, device=device
    )
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
    
    # Ensure verts has the right shape and type
    if not isinstance(verts, torch.Tensor):
        verts = torch.tensor(verts)
        
    verts = verts.to(device).to(torch.float)
    if len(verts.shape) == 2:
        verts = verts.unsqueeze(0)
        
    # Repeat points for each camera view
    sample_verts = verts.repeat(30, 1, 1)
    
    # Determine colors based on label
    if label is not None:
        # Use specific class color if label is provided
        color = torch.tensor(class_colors[label], device=device)
    else:
        # Default color if no label is provided
        color = torch.tensor([0.5, 0.5, 1.0], device=device)
        
    # Create color tensor for all points
    sample_colors = color.view(1, 1, 3).repeat(30, verts.shape[1], 1)
    
    # Create point cloud object
    point_cloud = pytorch3d.structures.Pointclouds(
        points=sample_verts,
        features=sample_colors
    ).to(device)
    
    # Render
    renderer = get_points_renderer(
        image_size=image_size, 
        background_color=background_color, 
        device=device
    )
    rend = renderer(point_cloud, cameras=cameras).cpu().numpy()
    
    # Convert to uint8 for saving
    rend = (rend * 255).astype(np.uint8)
    
    # Save as gif
    imageio.mimsave(path, rend, fps=15, loop=0)