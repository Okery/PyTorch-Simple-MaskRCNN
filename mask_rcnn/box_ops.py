import math

import torch


class BoxCoder:
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_box, proposal):
        height = proposal[:, 2] - proposal[:, 0]
        width = proposal[:, 3] - proposal[:, 1]
        ctr_y = proposal[:, 0] + 0.5 * height
        ctr_x = proposal[:, 1] + 0.5 * width

        gt_height = reference_box[:, 2] - reference_box[:, 0]
        gt_width = reference_box[:, 3] - reference_box[:, 1]
        gt_ctr_y = reference_box[:, 0] + 0.5 * gt_height
        gt_ctr_x = reference_box[:, 1] + 0.5 * gt_width

        dy = self.weights[0] * (gt_ctr_y - ctr_y) / height
        dx = self.weights[1] * (gt_ctr_x - ctr_x) / width
        dh = self.weights[2] * torch.log(gt_height / height)
        dw = self.weights[3] * torch.log(gt_width / width)

        delta = torch.stack((dy, dx, dh, dw), dim=1)
        return delta

    def decode(self, delta, box):
        dy = delta[:, 0] / self.weights[0]
        dx = delta[:, 1] / self.weights[1]
        dh = delta[:, 2] / self.weights[2]
        dw = delta[:, 3] / self.weights[3]

        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dw = torch.clamp(dw, max=self.bbox_xform_clip)

        height = box[:, 2] - box[:, 0]
        width = box[:, 3] - box[:, 1]
        ctr_y = box[:, 0] + 0.5 * height
        ctr_x = box[:, 1] + 0.5 * width

        pred_ctr_y = dy * height + ctr_y
        pred_ctr_x = dx * width + ctr_x
        pred_h = torch.exp(dh) * height
        pred_w = torch.exp(dw) * width

        ymin = pred_ctr_y - 0.5 * pred_h
        xmin = pred_ctr_x - 0.5 * pred_w
        ymax = pred_ctr_y + 0.5 * pred_h
        xmax = pred_ctr_x + 0.5 * pred_w

        target = torch.stack((ymin, xmin, ymax, xmax), dim=1)
        return target

    
def box_iou(box_a, box_b):
    tl = torch.max(box_a[:, None, :2], box_b[:, :2])
    br = torch.min(box_a[:, None, 2:], box_b[:, 2:])

    area_i = torch.prod(br - tl, 2) * (tl < br).all(2)
    area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], 1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], 1)
    
    return area_i / (area_a[:, None] + area_b - area_i)


def process_box(box, score, image_shape, min_size):
    box[:, [0, 2]] = box[:, [0, 2]].clamp(0, image_shape[0]) 
    box[:, [1, 3]] = box[:, [1, 3]].clamp(0, image_shape[1]) 

    hs, ws = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    keep = torch.where((hs >= min_size) & (ws >= min_size))[0]
    box, score = box[keep], score[keep]
    return box, score


def nms(box, score, threshold):
    try:
        from . import _C
        if box.shape[0] == 0:
            return torch.empty((0,), device=box.device, dtype=torch.int64)
        return _C.nms(box, score, threshold)
    except ImportError:
        y1, x1, y2, x2 = box.split(1, dim=1)
        t_box = torch.stack((x1, y1, x2, y2), dim=1)
        return torch.ops.torchvision.nms(t_box, score, threshold)
    


'''
# too slow
def nms(box, nms_thresh):
    idx = torch.arange(box.size(0))
    
    keep = []
    while idx.size(0) > 0:
        keep.append(idx[0].item())
        head_box = box[idx[0], None, :]
        remain = torch.where(box_iou(head_box, box[idx]) <= nms_thresh)[1]
        idx = idx[remain]
    
    return keep
'''