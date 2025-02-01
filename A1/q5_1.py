import torch
import pytorch3d 
import numpy as np
import imageio
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import FoVPerspectiveCameras, PointLights
from starter.utils import get_device, get_points_renderer, unproject_depth_image
from starter.render_generic import load_rgbd_data

def create_point_clouds(device=None):
   """Creates three point clouds from two RGB-D images - individual views and combined."""
   data = load_rgbd_data()
   device = get_device() if device is None else device

   image1, mask1, depth1 = [torch.Tensor(data[k]) for k in ["rgb1", "mask1", "depth1"]]
   image2, mask2, depth2 = [torch.Tensor(data[k]) for k in ["rgb2", "mask2", "depth2"]]
   
   points1, colors1 = unproject_depth_image(image1, mask1, depth1, data["cameras1"])
   points2, colors2 = unproject_depth_image(image2, mask2, depth2, data["cameras2"])

   vertices1 = torch.Tensor(points1).to(device).unsqueeze(0)
   features1 = torch.Tensor(colors1).to(device).unsqueeze(0)
   vertices2 = torch.Tensor(points2).to(device).unsqueeze(0)
   features2 = torch.Tensor(colors2).to(device).unsqueeze(0)
   vertices3 = torch.Tensor(torch.cat((points1, points2), 0)).to(device).unsqueeze(0)
   features3 = torch.Tensor(torch.cat((colors1, colors2), 0)).to(device).unsqueeze(0)

   point_cloud1 = Pointclouds(points=vertices1, features=features1)
   point_cloud2 = Pointclouds(points=vertices2, features=features2)
   point_cloud3 = Pointclouds(points=vertices3, features=features3)

   return point_cloud1, point_cloud2, point_cloud3

def render_point_cloud(point_cloud, render_size=256, bg_color=(1, 1, 1), output_dir="output/", output_name="point_cloud.gif",  device=None, animation_fps=15, rotation_step=5, camera_distance=7, camera_elevation=0, flip_orientation=True):
   """Renders a 360-degree animation of a point cloud with configurable camera parameters."""
   device = get_device() if device is None else device
   renderer = get_points_renderer(image_size=render_size, background_color=bg_color)
   
   initial_angle = torch.Tensor([0, 0, np.pi if flip_orientation else 0])
   rotation = pytorch3d.transforms.euler_angles_to_matrix(initial_angle, "XYZ")
   frames = []

   for angle in range(-180, 180, rotation_step):
       camera_position, translation = pytorch3d.renderer.look_at_view_transform(
           dist=camera_distance, 
           elev=camera_elevation, 
           azim=angle
       )
       camera = FoVPerspectiveCameras(
           R=camera_position @ rotation, 
           T=translation, 
           device=device
       )
       lights = PointLights(location=[[0, 0, -3]], device=device)

       frame = renderer(point_cloud, cameras=camera, lights=lights)
       frame = frame.cpu().numpy()[0, ..., :3]
       frame = (frame * 255).astype(np.uint8)
       frames.append(frame)

   imageio.mimsave(output_dir + output_name, frames, fps=animation_fps)

def main():
    point_cloud1, point_cloud2, point_cloud3 = create_point_clouds()
    render_point_cloud(point_cloud1, render_size=1024, output_name="q5_1_1.gif")
    render_point_cloud(point_cloud2, render_size=1024, output_name="q5_1_2.gif") 
    render_point_cloud(point_cloud3, render_size=1024, output_name="q5_1_3.gif")

if __name__ == "__main__":
    main()