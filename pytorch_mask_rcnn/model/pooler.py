import math

import torch
 

class MultiScaleRoIAlign:
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
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.scales = None
        
    def setup_scales(self, x, image_size):
        if self.scales is not None:
            return
        
        h, w = image_size
        grid_sizes = [tuple(feat.shape[2:]) for feat in x]
        scales = []
        for g in grid_sizes:
            h_scale = 2 ** int(math.log2(g[0] / h))
            w_scale = 2 ** int(math.log2(g[1] / w))
            assert h_scale == w_scale
            scales.append(h_scale)
        self.scales = scales
        
    def __call__(self, x, boxes, image_size):
        """
        Arguments:
            x (List[Tensor])
            boxes (List[Tensor])
            image_size (Torch.Size([H, W]))

        Returns:
            results (List[Tensor])
        
        """
        dtype = x[0].dtype
        ids = torch.cat([b.new_full((b.shape[0], 1), i) for i, b in enumerate(boxes)])
        boxes = torch.cat(boxes)
        rois = torch.cat((ids, boxes), dim=1).to(dtype)
        
        self.setup_scales(x, image_size)
        #print(x[0].dtype, rois.dtype)
        results = [
            roi_align(feature, rois, self.output_size, scale, self.sampling_ratio)
            for feature, scale in zip(x, self.scales)
        ]
        return results
    
    
def roi_align(x, rois, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    if len(rois) == 0:
        return rois.new_zeros((0, x.shape[1], output_size[0], output_size[1]))
    
    if torch.__version__ >= '1.5.0':
        return torch.ops.torchvision.roi_align(x, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned)
    return torch.ops.torchvision.roi_align(x, rois, spatial_scale, output_size[0], output_size[1], sampling_ratio)