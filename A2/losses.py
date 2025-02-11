import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	voxel_src = torch.sigmoid(voxel_src)
	loss = F.binary_cross_entropy(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	src_nn = knn_points(point_cloud_src, point_cloud_tgt, K=1)
	tgt_nn = knn_points(point_cloud_tgt, point_cloud_src, K=1)

	src_distances = src_nn.dists[..., 0]
	tgt_distances = tgt_nn.dists[..., 0]

	loss_chamfer = src_distances.mean() + tgt_distances.mean()

	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian