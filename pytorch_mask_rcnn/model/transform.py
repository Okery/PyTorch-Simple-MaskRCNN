import math
import random

import torch
import torch.nn.functional as F
from torch import nn


class Transformer(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__()
        if isinstance(min_size, int):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        
    def forward(self, images, targets):
        images = [img for img in images]
        targets = [tgt for tgt in targets] if targets is not None else None
            
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else None
            
            image = self.normalize(image)
            image, target = self.resize(image, target)
            
            images[i] = image
            if target is not None:
                targets[i] = target
                
        image_shapes = [img.shape[1:] for img in images]
        images = self.batched_images(images)
        return images, targets, image_shapes

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        orig_image_shape = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        
        if self.training:
            size = random.choice(self.min_size)
        else:
            size = max(self.min_size)
            
        scale_factor = min(size / min_size, self.max_size / max_size)
        image = F.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target
        
        box = target['boxes']
        # inplace operation, so no need of "target['boxes'] = box"
        box[:, [0, 2]] *= image.shape[2] / orig_image_shape[1]
        box[:, [1, 3]] *= image.shape[1] / orig_image_shape[0]
        
        if 'masks' in target:
            mask = target['masks']
            mask = F.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
            target['masks'] = mask
            
        return image, target
    
    def batched_images(self, images, stride=32):
        max_size = tuple(max(s) for s in zip(*(img.shape[1:] for img in images)))
        batch_size = tuple(math.ceil(m / stride) * stride for m in max_size)

        batch_shape = (len(images), 3,) + batch_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:, :img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs
    
    def postprocess(self, results, image_shapes, orig_image_shapes):
        for i, (res, im_s, o_im_s) in enumerate(zip(results, image_shapes, orig_image_shapes)):
            boxes = res['boxes']
            boxes[:, [0, 2]] *= o_im_s[1] / im_s[1]
            boxes[:, [1, 3]] *= o_im_s[0] / im_s[0]

            if 'masks' in res:
                masks = res['masks']
                masks = paste_masks_in_image(masks, boxes, 1, o_im_s)
                res['masks'] = masks
            
        return results


def expand_detection(mask, box, padding):
    M = mask.shape[-1]
    scale = (M + 2 * padding) / M
    padded_mask = torch.nn.functional.pad(mask, (padding,) * 4)
    
    w_half = (box[:, 2] - box[:, 0]) * 0.5
    h_half = (box[:, 3] - box[:, 1]) * 0.5
    x_c = (box[:, 2] + box[:, 0]) * 0.5
    y_c = (box[:, 3] + box[:, 1]) * 0.5

    w_half = w_half * scale
    h_half = h_half * scale

    box_exp = torch.zeros_like(box)
    box_exp[:, 0] = x_c - w_half
    box_exp[:, 2] = x_c + w_half
    box_exp[:, 1] = y_c - h_half
    box_exp[:, 3] = y_c + h_half
    return padded_mask, box_exp.to(torch.int64)


def paste_masks_in_image(mask, box, padding, image_shape):
    mask, box = expand_detection(mask, box, padding)
    
    N = mask.shape[0]
    size = (N,) + tuple(image_shape)
    im_mask = torch.zeros(size, dtype=mask.dtype, device=mask.device)
    for m, b, im in zip(mask, box, im_mask):
        b = b.tolist()
        w = max(b[2] - b[0], 1)
        h = max(b[3] - b[1], 1)
        
        m = F.interpolate(m[None, None], size=(h, w), mode='bilinear', align_corners=False)[0][0]

        x1 = max(b[0], 0)
        y1 = max(b[1], 0)
        x2 = min(b[2], image_shape[1])
        y2 = min(b[3], image_shape[0])

        im[y1:y2, x1:x2] = m[(y1 - b[1]):(y2 - b[1]), (x1 - b[0]):(x2 - b[0])]
    return im_mask