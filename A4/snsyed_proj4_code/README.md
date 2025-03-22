# 16-825 Assignment 4 submitted by Shahram Najam Syed (snsyed)

To execute the question solutions, activate the conda environment and navigate to the solution folder:

```
conda activate learning3d
cd <path_to_solution_folder>
```

The submitted report can be found over here: https://www.cs.cmu.edu/afs/andrew.cmu.edu/course/16/825/www/projects/snsyed/proj4/


### 1.1.5 Perform Splatting (5 points)

```
python volume_rendering_main.py --config-name=box
```

### 1.4. Point sampling (5 points)
```
python volume_rendering_main.py --config-name=box
```

### 1.5. Volume rendering (20 points)
```
python volume_rendering_main.py --config-name=box
```

### 2.3. Visualization
```
python volume_rendering_main.py --config-name=train_box
```
### 3. Optimizing a Neural Radiance Field (NeRF) (20 points)
```
python volume_rendering_main.py --config-name=nerf_lego
```

### 4.1 View Dependence (10 points)
```
python volume_rendering_main.py --config-name=nerf_materials
python volume_rendering_main.py --config-name=nerf_materials_high_res
```

### 4.2 Coarse/Fine Sampling (10 points) (Extra Credit)
```
python volume_rendering_main.py --config-name=nerf_materials
python volume_rendering_main.py --config-name=nerf_lego
```

### 5. Sphere Tracing (10 points)
```
python -m surface_rendering_main --config-name=torus_surface
```

### 6. Optimizing a Neural SDF (15 points)
```
python -m surface_rendering_main --config-name=points_surface
```

### 7. VolSDF (15 points)
```
python -m surface_rendering_main --config-name=volsdf_surface
```

### 8.1. Render a Large Scene with Sphere Tracing (10 points)
```
python -m surface_rendering_main --config-name=complex_scene
```

### 8.2 Fewer Training Views (10 points) (Extra Credit)
```
python -m surface_rendering_main --config-name=volsdf_surface
python volume_rendering_main.py --config-name=nerf_lego
```

### 8.3 Alternate SDF to Density Conversions (10 points) (Extra Credit)
```
python -m surface_rendering_main --config-name=complex_scene
```