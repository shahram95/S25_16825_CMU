from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # TODO:
            # self.decoder =
            self.fc = nn.Sequential(nn.Linear(512, 2048), nn.ReLU())
            self.layer1 = nn.Sequential(nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                                        nn.BatchNorm3d(128),
                                        nn.ReLU(inplace=True)
                                        )
            self.layer2 = nn.Sequential(nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                                        nn.BatchNorm3d(64),
                                        nn.ReLU(inplace=True)
                                        )
            self.layer3 = nn.Sequential(nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True)
                                        )
            self.layer4 = nn.Sequential(nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, padding=1, bias=False),
                                        nn.BatchNorm3d(8),
                                        nn.ReLU(inplace=True)
                                        )
            self.layer5 = nn.Sequential(nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
                                        nn.Sigmoid()
                                        )             
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),

                nn.Linear(1024,2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),

                nn.Linear(2048, self.n_point*3),
                nn.Tanh()
            )
                         
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            n_vertices = mesh_pred.verts_packed().shape[0]
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),

                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True),

                nn.Linear(2048, n_vertices*3),
                nn.Tanh()
            )

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images 

        # call decoder
        if args.type == "vox":
            # TODO:
            # voxels_pred =    
            x = self.fc(encoded_feat)
            x = x.view(-1, 256, 2, 2, 2)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            voxels_pred = self.layer5(x)         
            return voxels_pred

        elif args.type == "point":
            # TODO:
            point_features = self.decoder(encoded_feat)
            pointclouds_pred = point_features.view(B, self.n_point, 3)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)             
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred
        
class ParametricPointDecoder(nn.Module):
    def __init__(self, feature_size, hidden_sizes=[512, 256, 128], output_size=3):
        super(ParametricPointDecoder, self).__init__()
        layers = []
        input_size = feature_size + 2  # Concatenate image features with 2D (x, y) coordinates
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, image_features, coords_2d):
        # image_features: (B, feature_size)
        # coords_2d: (B, 2)
        x = torch.cat([image_features, coords_2d], dim=-1)
        coords_3d = self.mlp(x)
        return coords_3d

class ImplicitOccupancyDecoder(nn.Module):
    def __init__(self, feature_size, hidden_sizes=[512, 256, 128, 64], output_size=1):
        super(ImplicitOccupancyDecoder, self).__init__()
        layers = []
        input_size = feature_size + 3  # Concatenate image features with 3D (x, y, z) coordinates
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, output_size))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, image_features, coords):
        # image_features: (B, feature_size)
        # coords: (B, 3)
        x = torch.cat([image_features, coords], dim=-1)
        occupancy = self.mlp(x)
        return occupancy