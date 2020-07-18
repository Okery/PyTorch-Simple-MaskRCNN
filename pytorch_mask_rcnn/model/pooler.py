import math
import torch

from .utils import roi_align


class RoIAlign:
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN
    
    """
    
    def __init__(self, output_size, sampling_ratio):
        """
        Arguments:
            output_size (Tuple[int, int]): the size of the output after the cropping
                is performed, as (height, width)
            sampling_ratio (int): number of sampling points in the interpolation grid
                used to compute the output value of each pooled output bin. If > 0,
                then exactly sampling_ratio x sampling_ratio grid points are used. If
                <= 0, then an adaptive number of grid points are used (computed as
                ceil(roi_width / pooled_w), and likewise for height). Default: -1
        """
        
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
        """
        Arguments:
            feature (Tensor[N, C, H, W])
            proposal (Tensor[K, 4])
            image_shape (Torch.Size([H, W]))

        Returns:
            output (Tensor[K, C, self.output_size[0], self.output_size[1]])
        
        """
        idx = proposal.new_full((proposal.shape[0], 1), 0)
        roi = torch.cat((idx, proposal), dim=1)
        
        self.setup_scale(feature.shape[-2:], image_shape)
        return roi_align(feature.to(roi), roi, self.spatial_scale, self.output_size[0], self.output_size[1], self.sampling_ratio)