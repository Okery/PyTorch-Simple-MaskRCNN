import os
from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import resnet
from torchvision.ops import misc

from .utils import IntermediateLayerGetter
from .path_aggregation_network import PathAggregationNetwork

    
class BackboneWithPANet(nn.Sequential):
    """
    Adds a PANet on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the PANet.
    Attributes:
        out_channels (int): the number of channels in the PANet
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        self.out_channels = out_channels
        
        d = OrderedDict()
        d['body'] = IntermediateLayerGetter(backbone, return_layers=return_layers)
        d['pan'] = PathAggregationNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        super().__init__(d)
        

def resnet_pan_backbone(backbone_name, pretrained, ckpt_dir):
    if ckpt_dir is None:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
    else:
        backbone = resnet.__dict__[backbone_name](norm_layer=misc.FrozenBatchNorm2d)
    
    if ckpt_dir is not None and pretrained:
        weights = {
            'resnet50': 'resnet50-19c8e357.pth',
        }
        msd = torch.load(os.path.join(ckpt_dir, weights[backbone_name]))
        backbone.load_state_dict(msd)
        
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithPANet(backbone, return_layers, in_channels_list, out_channels)