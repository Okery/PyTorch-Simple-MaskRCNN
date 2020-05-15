import torch


class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between 
            M ground-truth boxes and N predicted boxes.

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """
        
        value, matched_idx = iou.max(dim=0)
        label = torch.full((iou.shape[1],), -1, dtype=torch.float, device=iou.device) 
        
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
    def __init__(self, sizes, aspect_ratios):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, size, dtype, device):
        size = torch.tensor([size], dtype=dtype, device=device)
        aspect_ratios = torch.tensor(self.aspect_ratios, dtype=dtype, device=device)
        w_ratios = torch.sqrt(aspect_ratios)
        h_ratios = 1 / w_ratios

        ws = w_ratios * size
        hs = h_ratios * size

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()
    
    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None:
            return

        cell_anchors = [
            self.generate_anchors(size, dtype, device)
            for size in self.sizes
        ]
        self.cell_anchors = cell_anchors
        
    def grid_anchors(self, grid_sizes, strides):
        dtype, device = self.cell_anchors[0].dtype, self.cell_anchors[0].device
        
        anchors_all_level = []
        for size, stride, base_anchors in zip(grid_sizes, strides, self.cell_anchors):
            shift_x = torch.arange(0, size[1], dtype=dtype, device=device) * stride[1]
            shift_y = torch.arange(0, size[0], dtype=dtype, device=device) * stride[0]

            y, x = torch.meshgrid(shift_y, shift_x)
            x = x.reshape(-1)
            y = y.reshape(-1)
            shift = torch.stack((x, y, x, y), dim=1).reshape(-1, 1, 4)

            anchor = (shift + base_anchors).reshape(-1, 4)
            anchors_all_level.append(anchor)
            
        return torch.cat(anchors_all_level)
        
    def cached_grid_anchors(self, grid_sizes, strides):
        key = grid_sizes + strides
        if key in self._cache:
            return self._cache[key]
        anchor = self.grid_anchors(grid_sizes, strides)
        
        if len(self._cache) >= 3:
            self._cache.clear()
        self._cache[key] = anchor
        return anchor

    def __call__(self, features, image_size):
        dtype, device = features[0].dtype, features[0].device
        grid_sizes = tuple(tuple(f.shape[2:]) for f in features)
        strides = tuple((int(image_size[0] / g[0]), int(image_size[1] / g[1])) for g in grid_sizes)
        
        self.set_cell_anchors(dtype, device)
        anchor_per_image = self.cached_grid_anchors(grid_sizes, strides)
        return anchor_per_image