# 16-825 Assignment 4 submitted by Shahram Najam Syed (snsyed)

To execute the question solutions, activate the conda environment and navigate to the solution folder:

```
conda activate learning3d
cd <path_to_solution_folder>
```

The submitted report can be found over here: https://www.cs.cmu.edu/afs/andrew.cmu.edu/course/16/825/www/projects/snsyed/proj4/


### 1.1 3D Gaussian Rasterization (35 points)

```
python render.py
```

### 1.2 Training 3D Gaussian Representations (15 points)
```
python train.py
```

### 1.3.1 Rendering Using Spherical Harmonics (10 Points)
```
python render.py
```

### 1.3.2 Training On a Harder Scene (10 Points)
```
python train_harder_scene.py --gaussians_per_splat 4000
```
### 2.1 SDS Loss + Image Optimization (20 points)
```
python Q21_image_optimization.py --prompt="a hamburger" --sds_guidance=0
python Q21_image_optimization.py --prompt="a hamburger" --sds_guidance=1 --postfix="_guidance"
python Q21_image_optimization.py --prompt="a standing corgi dog" --sds_guidance=0
python Q21_image_optimization.py --prompt="a standing corgi dog" --sds_guidance=1 --postfix="_guidance"
python Q21_image_optimization.py --prompt="piano playing shiba inu dog" --sds_guidance=0
python Q21_image_optimization.py --prompt="piano playing shiba inu dog" --sds_guidance=1 --postfix="_guidance"
python Q21_image_optimization.py --prompt="a water type dragon pokemon" --sds_guidance=0
python Q21_image_optimization.py --prompt="a water type dragon pokemon" --sds_guidance=1 --postfix="_guidance"
python Q21_image_optimization.py --prompt="a standing baby yoda" --sds_guidance=0
python Q21_image_optimization.py --prompt="a standing baby yoda" --sds_guidance=1 --postfix="_guidance"
```

### 2.2 Texture Map Optimization for Mesh (15 points)
```
python Q22_mesh_optimization.py --prompt="A cow wearing batman textured costume"
python Q22_mesh_optimization.py --prompt="A cow made of shiny lava obsidian"
python Q22_mesh_optimization.py --prompt="A cow with cosmic clouds patterns"
```

### 2.3 NeRF Optimization (15 points)
```
python Q23_nerf_optimization.py --prompt="A standing corgi dog" --iters=4100 --lambda_entropy=0.0005 --lambda_orient=0.01 --latent_iter_ratio=0.2
python Q23_nerf_optimization.py --prompt="A blue coffee mug" --iters=4100 --lambda_entropy=0.0005 --lambda_orient=0.01 --latent_iter_ratio=0.2
python Q23_nerf_optimization.py --prompt="a sea turtle" --iters=4100 --lambda_entropy=0.0005 --lambda_orient=0.01 --latent_iter_ratio=0.2
python Q23_nerf_optimization.py --prompt="a feasting squirrel" --iters=4100 --lambda_entropy=0.0005 --lambda_orient=0.01 --latent_iter_ratio=0.2
```