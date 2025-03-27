import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        # Classification network - MLP for class prediction
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Regularization
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Regularization
            
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # Transpose to the expected format for Conv1d: [batch, channels, points]
        points = points.transpose(2, 1)
        
        # Extract features using the encoder
        features = self.encoder(points)
        
        # Global max pooling to get a global feature vector
        # This creates a permutation-invariant function on the point set
        global_features = torch.max(features, dim=2, keepdim=False)[0]
        
        # Get classification scores using the classifier network
        output = self.classifier(global_features)
        
        return output



# ------ TO DO ------
def conv_bn(in_channels, out_channels):
    """Helper function to create a convolutional layer followed by batch normalization."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )

class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # Local feature extraction network
        self.local_features = nn.Sequential(
            conv_bn(3, 64),
            conv_bn(64, 64)
        )
        
        # Global feature extraction network
        self.global_features = nn.Sequential(
            conv_bn(64, 128),
            conv_bn(128, 1024)
        )
        
        # Segmentation network that processes combined local and global features
        self.segmentation_network = nn.Sequential(
            conv_bn(1088, 512),  # 1088 = 64 (local) + 1024 (global)
            conv_bn(512, 256),
            conv_bn(256, 128),
            nn.Conv1d(128, num_seg_classes, 1)  # Final prediction layer
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        # Get batch size and number of points
        B, N, _ = points.size()
        
        # Transpose for 1D convolution [B, 3, N]
        points = points.transpose(1, 2)
        
        # Extract local point features
        local_feat = self.local_features(points)  # [B, 64, N]
        
        # Extract global features
        x = self.global_features(local_feat)  # [B, 1024, N]
        
        # Get global feature with max pooling
        global_feat = torch.max(x, dim=2, keepdim=True)[0]  # [B, 1024, 1]
        
        # Expand global feature to all points
        global_feat_expanded = global_feat.expand(-1, -1, N)  # [B, 1024, N]
        
        # Concatenate local and global features
        combined_feat = torch.cat([local_feat, global_feat_expanded], dim=1)  # [B, 1088, N]
        
        # Process through segmentation network
        segmentation_logits = self.segmentation_network(combined_feat)  # [B, num_seg_classes, N]
        
        # Transpose back to get [B, N, num_seg_classes]
        segmentation_logits = segmentation_logits.transpose(1, 2)
        
        return segmentation_logits