import math

import torch
 

class RoIAlign:
    def __init__(self, output_size, sampling_ratio):
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.spatial_scale = None
        
    def setup_scale(self, feature_shape, image_shape):
        if self.spatial_scale is not None:
            return
        
        possible_scales = []
        for s1, s2 in zip(feature_shape, image_shape):
            scale = 2 ** int(math.log2(s1 / s2))
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        self.spatial_scale = possible_scales[0]
        
    def __call__(self, feature, proposal, image_shape):
        idx = proposal.new_full((proposal.shape[0], 1), 0)
        roi = torch.cat((idx, proposal), dim=1)
        
        self.setup_scale(feature.shape[-2:], image_shape)
        return torch.ops.torchvision.roi_align(feature, roi, self.spatial_scale, self.output_size[0], self.output_size[1], self.sampling_ratio)