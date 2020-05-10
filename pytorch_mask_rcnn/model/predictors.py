from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

    
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)
        
    def forward(self, features):
        for i in range(len(features)):
            feat = features[i]
            feat = feat.flatten(start_dim=1)
            features[i] = F.relu(self.fc1(feat))
            
        x = torch.max(torch.stack(features), dim=0)[0] # max fusion operation
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta.reshape(score.shape[0], -1, 4)  


class MaskRCNNPredictor(nn.Module):
    def __init__(self, in_channels, layers, dim_reduced,
                 num_classes, position, resolution):
        """
        Args:
            featmap_names (List[str]): the names of the feature maps that will be used.
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
            layers (Tuple[int])
            position (int)
        """
        super().__init__()
        self.main_path = FCN(in_channels, layers, dim_reduced, num_classes)
        self.complementary = FusionBlock(layers, position, resolution)
        self.num_classes = num_classes
        self.position = position
                
    def forward(self, features):
        """
        Args:
            features (List[Tensor]): mask feature maps for each level. They are assumed to have
                all the same shapes.
        """
        feat = features[0]
        if len(feat) == 0:
            resolution = feat.shape[2] * 2
            return feat.new_zeros(0, self.num_classes, resolution, resolution)
        
        for idx, module in enumerate(self.main_path.values(), 1):
            if idx == 1:
                for i in range(len(features)):
                    features[i] = F.relu(module(features[i]))
                x = torch.max(torch.stack(features), dim=0)[0] # max fusion operation
                continue
            
            x = F.relu(module(x))
            if idx == self.position - 1:
                y = self.complementary(F.relu(x))
                resolution = round(y.shape[-1] ** 0.5)
                y = y.reshape(-1, 1, resolution, resolution)
        x += y # sum fusion operation
        return x
        
        
class FCN(nn.ModuleDict):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            next_feature = layer_features
        
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        
        super().__init__(d)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')  
                

class FusionBlock(nn.Module):
    def __init__(self, layers, position, resolution):
        super().__init__()
        d = OrderedDict()
        in_channels = layers[position - 2]
        d['mask_conv{}_fc'.format(position)] = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        d['relu{}'.format(position)] = nn.ReLU(inplace=True)
        
        dim_reduced = in_channels // 2
        d['mask_conv{}_fc'.format(position + 1)] = nn.Conv2d(in_channels, dim_reduced, 3, 1, 1)
        d['relu{}'.format(position + 1)] = nn.ReLU(inplace=True)
        
        self.conv_layers = nn.Sequential(d)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu') 
                
        self.mask_fc = nn.Linear(dim_reduced * resolution ** 2, (resolution * 2) ** 2)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.mask_fc(x)
        return x