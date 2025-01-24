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