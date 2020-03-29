import torch


class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        value, matched_idx = iou.max(dim=0)
        label = torch.full((iou.shape[1],), -1, device=iou.device) 
        
        label[value >= self.high_threshold] = 1
        label[value < self.low_threshold] = 0
        
        if self.allow_low_quality_matches:
            highest_quality = iou.max(dim=1)[0]
            gt_pred_pairs = torch.where(iou == highest_quality[:, None])[1]
            label[gt_pred_pairs] = 1

        return label, matched_idx
    

class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = torch.where(label == 1)[0]
        negative = torch.where(label == 0)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)

        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx
    

class AnchorGenerator:
    def __init__(self, sizes, ratios):
        self.sizes = sizes
        self.ratios = ratios
        
        self.cell_anchor = None
        self._cache = {}
        
    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return 
        sizes = torch.tensor(self.sizes, dtype=dtype, device=device)
        ratios = torch.tensor(self.ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios

        hs = (sizes[:, None] * h_ratios[None, :]).view(-1)
        ws = (sizes[:, None] * w_ratios[None, :]).view(-1)

        self.cell_anchor = torch.stack([-hs, -ws, hs, ws], dim=1) / 2
        
    def grid_anchor(self, grid_size, stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        shift_y = torch.arange(0, grid_size[0], dtype=dtype, device=device) * stride[0]
        shift_x = torch.arange(0, grid_size[1], dtype=dtype, device=device) * stride[1]

        y, x = torch.meshgrid(shift_y, shift_x)
        y = y.reshape(-1)
        x = x.reshape(-1)
        shift = torch.stack((y, x, y, x), dim=1).reshape(-1, 1, 4)

        anchor = (shift + self.cell_anchor).reshape(-1, 4)
        return anchor
        
    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride
        if key in self._cache:
            return self._cache[key]
        anchor = self.grid_anchor(grid_size, stride)
        
        if len(self._cache) >= 3:
            self._cache.clear()
        self._cache[key] = anchor
        return anchor

    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        grid_size = tuple(feature.shape[-2:])
        stride = tuple(int(i / g) for i, g in zip(image_size, grid_size))
        
        self.set_cell_anchor(dtype, device)
        
        anchor = self.cached_grid_anchor(grid_size, stride)
        return anchor