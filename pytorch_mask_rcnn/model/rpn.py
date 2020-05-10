import torch
import torch.nn.functional as F
from torch import nn

from . import box_ops
from .utils import Matcher, BalancedPositiveNegativeSampler


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1)
        
        for l in self.children():
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
            
    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head, 
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction, reg_weights,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        
        self.anchor_generator = anchor_generator
        self.head = head
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = box_ops.BoxCoder(reg_weights)
        
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1
       
    @property
    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']
    
    @property
    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']
    
    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, dim=1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
            top_n_idx = ob.topk(pre_nms_top_n, dim=1)[1]
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def create_proposals(self, anchor, objectness, bbox_deltas, image_shapes, num_anchors_per_level):
        N, device = objectness.shape[0], objectness.device # number of images
        anchors = anchor.repeat(N, 1, 1)
        
        levels = torch.cat([
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ])
        levels = levels.repeat(N, 1)

        # select top n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        #print('top_n_idx', tuple(top_n_idx.shape))

        batch_idx = torch.arange(N, device=device)[:, None]

        anchors = anchors[batch_idx, top_n_idx]
        objectness = objectness[batch_idx, top_n_idx]
        bbox_deltas = bbox_deltas[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]

        proposals = []
        for idx in range(N):
            score = objectness[idx]
            lvl = levels[idx]
            
            proposal = self.box_coder.decode(bbox_deltas[idx], anchors[idx])
            proposal = box_ops.clip_boxes_to_image(proposal, image_shapes[idx])
            keep = box_ops.remove_small_boxes(proposal, self.min_size)
            proposal, score, lvl = proposal[keep], score[keep], lvl[keep]
            
            keep = box_ops.batched_nms(proposal, score, lvl, self.nms_thresh)[:self.post_nms_top_n] 
            proposal = proposal[keep]
            proposals.append(proposal)
        return proposals
    
    def compute_loss(self, objectness, bbox_deltas, anchor, targets):
        objectness_loss = []
        box_loss = []
        for score, delta, target in zip(objectness, bbox_deltas, targets):
            gt_boxes = target['boxes']
            iou = box_ops.box_iou(gt_boxes, anchor)
            label, matched_idx = self.proposal_matcher(iou)
        
            pos_idx, neg_idx = self.fg_bg_sampler(label)
            idx = torch.cat((pos_idx, neg_idx))
            regression_target = self.box_coder.encode(gt_boxes[matched_idx[pos_idx]], anchor[pos_idx])

            objectness_loss_per_image = F.binary_cross_entropy_with_logits(score[idx], label[idx])
            box_loss_per_image = F.l1_loss(delta[pos_idx], regression_target, reduction='sum') / idx.numel()
            
            objectness_loss.append(objectness_loss_per_image)
            box_loss.append(box_loss_per_image)

        return sum(objectness_loss), sum(box_loss)
        
    def forward(self, features, batch_shape, image_shapes, targets=None):
        anchor = self.anchor_generator(features, batch_shape)
        
        objectness, bbox_deltas = self.head(features)
        N = len(image_shapes)
        num_anchors_per_level = [o[0].numel() for o in objectness]
        #print(anchor[:3])
        #print(anchor[num_anchors_per_level[0]:num_anchors_per_level[0]+3])
        #print('num_anchors_per_level', num_anchors_per_level)
        objectness = torch.cat([o.permute(0, 2, 3, 1).flatten(start_dim=1) for o in objectness], dim=1)
        bbox_deltas = torch.cat([d.permute(0, 2, 3, 1).reshape(N, -1, 4) for d in bbox_deltas], dim=1)

        proposals = self.create_proposals(anchor, objectness.detach(), bbox_deltas.detach(),
                                          image_shapes, num_anchors_per_level)
        losses = {}
        if self.training:
            objectness_loss, box_loss = self.compute_loss(objectness, bbox_deltas, anchor, targets)
            losses = dict(rpn_objectness_loss=objectness_loss, rpn_box_loss=box_loss)
        
        return proposals, losses