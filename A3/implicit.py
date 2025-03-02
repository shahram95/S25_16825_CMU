import torch
import torch.nn.functional as F
from torch import autograd
import math
from ray_utils import RayBundle
import torch.nn as nn

# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        self.mlp_xyz = MLPWithInputSkips(
            n_layers=cfg.n_layers_xyz,
            input_dim=embedding_dim_xyz,
            output_dim=cfg.n_hidden_neurons_xyz,
            skip_dim=embedding_dim_xyz,
            hidden_dim=cfg.n_hidden_neurons_xyz,
            input_skips=cfg.append_xyz
        )

        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1),
            torch.nn.ReLU()
        )
        self.feature_layer = torch.nn.Sequential(
            torch.nn.Linear(cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz),
            torch.nn.ReLU()
        )
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(cfg.n_hidden_neurons_xyz + embedding_dim_dir, cfg.n_hidden_neurons_dir),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid()
        )

    
    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points
        directions = ray_bundle.directions        
        points_embedding = self.harmonic_embedding_xyz(sample_points)        
        features = self.mlp_xyz(points_embedding, points_embedding)
        density = self.density_layer(features)
        point_features = self.feature_layer(features)
        directions_embedding = self.harmonic_embedding_dir(directions)
        directions_embedding = directions_embedding.unsqueeze(1).expand(-1, sample_points.shape[1], -1)
        color_input = torch.cat([point_features, directions_embedding], dim=-1)
        color = self.color_layer(color_input)
        
        return {
            'density': density,
            'feature': color
        }


class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # TODO (Q6): Implement Neural Surface MLP to output per-point SDF
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim

        self.mlp = MLPWithInputSkips(
            n_layers=cfg.n_layers_distance,
            input_dim=embedding_dim_xyz,
            output_dim=cfg.n_hidden_neurons_distance,
            skip_dim=embedding_dim_xyz,
            hidden_dim=cfg.n_hidden_neurons_distance,
            input_skips=cfg.append_distance
        )

        self.distance_head = torch.nn.Linear(cfg.n_hidden_neurons_distance, 1)

        with torch.no_grad():
            torch.nn.init.xavier_normal_(self.distance_head.weight)
            self.distance_head.bias.data.fill_(0.0)

        # TODO (Q7): Implement Neural Surface MLP to output per-point color
        self.color_network = torch.nn.Sequential(
            torch.nn.Linear(cfg.n_hidden_neurons_distance + 3, cfg.n_hidden_neurons_color),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.n_hidden_neurons_color, cfg.n_hidden_neurons_color),
            torch.nn.ReLU(),
            torch.nn.Linear(cfg.n_hidden_neurons_color, 3),
            torch.nn.Sigmoid()  # Constrain output to [0,1] for RGB values
        )

    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q6
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        points_embedded = self.harmonic_embedding_xyz(points)
        features = self.mlp(points_embedded, points_embedded)
        distance = self.distance_head(features)
        return distance

    def get_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        distance, features = self.get_distance(points)
        color_input = torch.cat([features, points], dim=-1)
        colors = self.color_network(color_input)
        return colors
    
    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        points_embedded = self.harmonic_embedding_xyz(points)
        features = self.mlp(points_embedded, points_embedded)
        distance = self.distance_head(features)
        color_input = torch.cat([features, points], dim=-1)
        colors = self.color_network(color_input)
        
        return distance, colors
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient

# class ComplexScene(torch.nn.Module):
#     def __init__(
#         self,
#         cfg,
#     ):
#         super().__init__()
        
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         self.dummy_param = nn.Parameter(torch.zeros(1, device=self.device))        
#         self.primitives = []
#         self.rotation_speed = 0.5  # Controls animation speed
        
#         # The Sun (central glowing sphere)
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([0.0, 0.0, 0.0], device=self.device),
#             "radius": 1.2,
#             "color": torch.tensor([1.0, 0.7, 0.2], device=self.device),
#             "emissive": True,
#             "id": "sun"
#         })
        
#         # Mercury
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([1.8, 0.0, 0.0], device=self.device),
#             "radius": 0.15,
#             "color": torch.tensor([0.7, 0.6, 0.5], device=self.device),
#             "orbit_radius": 1.8,
#             "orbit_speed": 4.0,
#             "id": "mercury"
#         })
        
#         # Venus with atmosphere
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([2.5, 0.0, 0.0], device=self.device),
#             "radius": 0.2,
#             "color": torch.tensor([0.9, 0.7, 0.4], device=self.device),
#             "orbit_radius": 2.5,
#             "orbit_speed": 3.0,
#             "id": "venus"
#         })
        
#         # Venus atmosphere
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([2.5, 0.0, 0.0], device=self.device),
#             "radius": 0.23,
#             "color": torch.tensor([1.0, 0.8, 0.6], device=self.device),
#             "orbit_radius": 2.5,
#             "orbit_speed": 3.0,
#             "opacity": 0.3,
#             "id": "venus_atmosphere"
#         })
        
#         # Earth
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([3.3, 0.0, 0.0], device=self.device),
#             "radius": 0.25,
#             "color": torch.tensor([0.2, 0.4, 0.8], device=self.device),
#             "orbit_radius": 3.3,
#             "orbit_speed": 2.5,
#             "id": "earth"
#         })
        
#         # Earth's Moon
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([3.3, 0.0, 0.4], device=self.device),
#             "radius": 0.07,
#             "color": torch.tensor([0.8, 0.8, 0.8], device=self.device),
#             "orbit_radius": 0.4,
#             "orbit_center": "earth",
#             "orbit_speed": 10.0,
#             "id": "moon"
#         })
        
#         # Mars
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([4.0, 0.0, 0.0], device=self.device),
#             "radius": 0.22,
#             "color": torch.tensor([0.9, 0.3, 0.2], device=self.device),
#             "orbit_radius": 4.0,
#             "orbit_speed": 2.0,
#             "id": "mars"
#         })
        
#         # Asteroid belt (multiple small rocks)
#         for i in range(40):
#             angle = i * 2 * math.pi / 40
#             radius_var = 4.8 + 0.4 * torch.rand(1).item()
#             size_var = 0.02 + 0.04 * torch.rand(1).item()
#             y_offset = 0.1 * torch.randn(1).item()
            
#             self.primitives.append({
#                 "type": "sphere",
#                 "center": torch.tensor([
#                     radius_var * math.cos(angle),
#                     y_offset,
#                     radius_var * math.sin(angle)
#                 ], device=self.device),
#                 "radius": size_var,
#                 "color": torch.tensor([0.6, 0.6, 0.5], device=self.device),
#                 "orbit_radius": radius_var,
#                 "orbit_speed": 1.5 + torch.rand(1).item(),
#                 "orbit_y_offset": y_offset,
#                 "id": f"asteroid_{i}"
#             })
        
#         # Jupiter (gas giant)
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([6.0, 0.0, 0.0], device=self.device),
#             "radius": 0.5,
#             "color": torch.tensor([0.8, 0.7, 0.6], device=self.device),
#             "orbit_radius": 6.0,
#             "orbit_speed": 1.0,
#             "id": "jupiter"
#         })
        
#         # Jupiter's bands (decorative stripes)
#         for i in range(5):
#             offset = 0.1 * (i - 2)
#             self.primitives.append({
#                 "type": "torus",
#                 "center": torch.tensor([6.0, offset, 0.0], device=self.device),
#                 "radii": torch.tensor([0.5, 0.04], device=self.device),
#                 "color": torch.tensor([0.9, 0.8, 0.7] if i % 2 == 0 else [0.7, 0.6, 0.5], device=self.device),
#                 "orbit_radius": 6.0,
#                 "orbit_speed": 1.0,
#                 "id": f"jupiter_band_{i}"
#             })
        
#         # Saturn
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([7.5, 0.0, 0.0], device=self.device),
#             "radius": 0.45,
#             "color": torch.tensor([0.9, 0.8, 0.6], device=self.device),
#             "orbit_radius": 7.5,
#             "orbit_speed": 0.8,
#             "id": "saturn"
#         })
        
#         # Saturn's rings
#         for i in range(3):
#             ring_radius = 0.65 + i * 0.15
#             self.primitives.append({
#                 "type": "torus",
#                 "center": torch.tensor([7.5, 0.0, 0.0], device=self.device),
#                 "radii": torch.tensor([ring_radius, 0.03], device=self.device),
#                 "color": torch.tensor([0.9, 0.85, 0.7], device=self.device),
#                 "orbit_radius": 7.5,
#                 "orbit_speed": 0.8,
#                 "id": f"saturn_ring_{i}"
#             })
        
#         # Uranus
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([8.7, 0.0, 0.0], device=self.device),
#             "radius": 0.35,
#             "color": torch.tensor([0.6, 0.8, 0.9], device=self.device),
#             "orbit_radius": 8.7,
#             "orbit_speed": 0.6,
#             "id": "uranus"
#         })
        
#         # Neptune
#         self.primitives.append({
#             "type": "sphere",
#             "center": torch.tensor([9.8, 0.0, 0.0], device=self.device),
#             "radius": 0.33,
#             "color": torch.tensor([0.2, 0.4, 0.9], device=self.device),
#             "orbit_radius": 9.8,
#             "orbit_speed": 0.5,
#             "id": "neptune"
#         })
        
#         # Background stars (small bright dots)
#         for i in range(50):
#             # Generate random positions in a spherical distribution
#             theta = 2 * math.pi * torch.rand(1).item()
#             phi = math.acos(2 * torch.rand(1).item() - 1)
#             radius = 15.0
            
#             # Convert to Cartesian coordinates
#             x = radius * math.sin(phi) * math.cos(theta)
#             y = radius * math.sin(phi) * math.sin(theta)
#             z = radius * math.cos(phi)
            
#             # Create a small bright star
#             star_color = torch.tensor([
#                 0.8 + 0.2 * torch.rand(1).item(),
#                 0.8 + 0.2 * torch.rand(1).item(),
#                 0.8 + 0.2 * torch.rand(1).item()
#             ], device=self.device)
            
#             self.primitives.append({
#                 "type": "sphere",
#                 "center": torch.tensor([x, y, z], device=self.device),
#                 "radius": 0.05 + 0.05 * torch.rand(1).item(),
#                 "color": star_color,
#                 "id": f"star_{i}"
#             })

#     def update_positions(self, time):
#         """Update positions of all objects based on their orbits"""
#         for primitive in self.primitives:
#             if "orbit_radius" in primitive:
#                 if "orbit_center" in primitive:
#                     # Find the center object
#                     for p in self.primitives:
#                         if p["id"] == primitive["orbit_center"]:
#                             center_pos = p["center"]
#                             break
#                 else:
#                     # Orbit around the sun
#                     center_pos = torch.tensor([0.0, 0.0, 0.0], device=self.device)
                
#                 # Calculate orbital position
#                 angle = (time * primitive["orbit_speed"] * self.rotation_speed) % (2 * math.pi)
#                 y_offset = primitive.get("orbit_y_offset", 0.0)
                
#                 # Update center position
#                 primitive["center"] = center_pos + torch.tensor([
#                     primitive["orbit_radius"] * math.cos(angle),
#                     y_offset,
#                     primitive["orbit_radius"] * math.sin(angle)
#                 ], device=self.device)

#     def forward(self, points, time=0.0):
#         """Compute the SDF value for each point"""
#         # Update positions based on time
#         self.update_positions(time)
        
#         batch_size = points.shape[0]
#         sdf = torch.ones(batch_size, 1, device=points.device) * 100  # Initialize with large values
        
#         for primitive in self.primitives:
#             if primitive["type"] == "sphere":
#                 center = primitive["center"]
#                 radius = primitive["radius"]
#                 dist = torch.norm(points - center, dim=1, keepdim=True) - radius
                
#             elif primitive["type"] == "torus":
#                 center = primitive["center"]
#                 radii = primitive["radii"]
#                 p = points - center
                
#                 # Torus SDF calculation
#                 x2z2 = torch.sum(p[:, [0, 2]]**2, dim=1, keepdim=True)
#                 dist = torch.sqrt(x2z2 + p[:, 1:2]**2 + radii[1]**2 - 2 * radii[0] * torch.sqrt(x2z2)) - radii[1]
                
#             else:
#                 # Skip unknown primitive types
#                 continue
            
#             # Apply union operation (minimum)
#             sdf = torch.minimum(sdf, dist)
            
#         return sdf
    
#     def get_color(self, points, time=0.0):
#         """Get the color of the closest primitive for each point"""
#         # Update positions based on time
#         self.update_positions(time)
        
#         batch_size = points.shape[0]
#         colors = torch.zeros(batch_size, 3, device=points.device)
        
#         # Find the closest primitive for each point
#         for i, point in enumerate(points):
#             min_dist = float('inf')
#             min_idx = 0
            
#             for j, primitive in enumerate(self.primitives):
#                 if primitive["type"] == "sphere":
#                     center = primitive["center"]
#                     radius = primitive["radius"]
#                     dist = torch.norm(point - center) - radius
                    
#                 elif primitive["type"] == "torus":
#                     center = primitive["center"]
#                     radii = primitive["radii"]
#                     p = point - center
                    
#                     x2z2 = torch.sum(p[[0, 2]]**2)
#                     dist = torch.sqrt(x2z2 + p[1]**2 + radii[1]**2 - 2 * radii[0] * torch.sqrt(x2z2)) - radii[1]
                    
#                 else:
#                     # Skip unknown primitive types
#                     continue
                
#                 if dist < min_dist:
#                     min_dist = dist
#                     min_idx = j
            
#             colors[i] = self.primitives[min_idx]["color"]
            
#         return colors

implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
    'complex_scene': ComplexScene,
}
