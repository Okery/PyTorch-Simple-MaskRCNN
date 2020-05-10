import torch
import torch.nn.functional as F
from torch import nn

from .pooler import MultiScaleRoIAlign, roi_align
from .utils import Matcher, BalancedPositiveNegativeSampler
from . import box_ops


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Args:
        class_logits (Tensor[N, L])
        box_regression (Tensor[N, L, 4])
        labels (List[Tensor[B]])
        regression_targets (List[Tensor[P, 4]])
            
    Returns:
        classification_loss (Tensor)
        box_reg_loss (Tensor)

    """
    labels = torch.cat(labels)
    regression_targets = torch.cat(regression_targets)
    pos_ids = torch.where(labels > 0)[0]
    
    classification_loss = F.cross_entropy(class_logits, labels)

    N = class_logits.shape[0]
    labels = labels[pos_ids]
    box_reg_loss = F.smooth_l1_loss(
        box_regression[pos_ids, labels],
        regression_targets,
        reduction='sum'
    ) / N

    return classification_loss, box_reg_loss


def maskrcnn_loss(mask_logits, proposals, matched_ids, labels, targets):
    """
    Args:
        mask_logis (Tensor[N, L, M, M]): L = num_classes, M = resolution (usually 28).
        proposals (List[Tensor[B, 4]])
        matched_ids (List[Tensor[B]])
        labels (List[Tensor[B]])
        targets (List[Dict])
        
    Returns:
        mask_loss (Tensor)

    """
    M = mask_logits.shape[-1]
    mask_targets = []
    for i in range(len(proposals)):
        proposal = proposals[i]
        matched_id = matched_ids[i][:, None].to(proposal)
        rois = torch.cat((matched_id, proposal), dim=1)
        gt_masks = targets[i]['masks'][:, None].to(rois)
        mask_targets.append(roi_align(gt_masks, rois, (M, M))[:, 0])

    mask_targets = torch.cat(mask_targets)
    labels = torch.cat(labels)
    idx = torch.arange(labels.shape[0], device=labels.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits[idx, labels], mask_targets)
    return mask_loss
    

class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        
        self.mask_roi_pool = None
        self.mask_predictor = None
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = box_ops.BoxCoder(reg_weights)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1
        
    @property
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
        
    def select_training_samples(self, proposals, targets):
        """
        Args:
            proposals (List[Tensor[B, 4]])
            targets (List[Dict])
            
        Returns:
            proposals (List[Tensor[B', 4]])
            matched_ids (List[Tensor[B']]): indices of matched gt boxes.
            labels (List[Tensor[B']]): classes' label, class = 0 means this proposal is background and thus is negative.
            regression_targets (List[Tensor[P, 4]]): For each image, only select positive samples' regression_target.
       
        """
        matched_ids, labels, regression_targets = [], [], []
        for i in range(len(proposals)):
            proposal = proposals[i]
            target = targets[i]
            
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            proposal = torch.cat((proposal, gt_boxes))

            iou = box_ops.box_iou(gt_boxes, proposal)
            pos_neg_label, matched_id = self.proposal_matcher(iou)
            pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
            idx = torch.cat((pos_idx, neg_idx))

            regression_target = self.box_coder.encode(gt_boxes[matched_id[pos_idx]], proposal[pos_idx])
            proposal = proposal[idx]
            matched_id = matched_id[idx]
            label = gt_labels[matched_id]
            num_pos = pos_idx.shape[0]
            label[num_pos:] = 0
            
            proposals[i] = proposal
            matched_ids.append(matched_id)
            labels.append(label)
            regression_targets.append(regression_target)
        
        return proposals, matched_ids, labels, regression_targets
    
    def fastrcnn_inference(self, class_logits, box_regression, proposals, image_shapes):
        """
        Args:
            class_logits (Tensor[N, L]): N = all boxes, L = num_classes.
            box_regression (Tensor[N, L, 4])
            proposals (List[Tensor[B, 4]])
            image_shapes (List[Tuple[H, W]])
            
        Returns:
            results (List[Dict]): keys of Dict: boxes, scores, labels.
            
        """
        N, num_classes = class_logits.shape
        
        device = class_logits.device
        pred_scores = F.softmax(class_logits, dim=1)[:, 1:] # delete background class
        box_regression = box_regression[:, 1:] # delete background class
        
        boxes_per_image = [p.shape[0] for p in proposals]
        pred_scores_list = pred_scores.split(boxes_per_image)
        box_regression_list = box_regression.split(boxes_per_image)
        
        results = []
        for i in range(len(proposals)):
            scores = pred_scores_list[i].flatten()
            box_deltas = box_regression_list[i].reshape(-1, 4)
            proposal = proposals[i].reshape(-1, 1, 4)
            proposal = proposal.repeat(1, num_classes - 1, 1).reshape(-1, 4)
            boxes = self.box_coder.decode(box_deltas, proposal)
            
            labels = torch.arange(1, num_classes, device=device).repeat(boxes_per_image[i])

            inds = scores >= self.score_thresh
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            
            boxes = box_ops.clip_boxes_to_image(boxes, image_shapes[i])
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            keep = box_ops.nms(boxes, scores, self.nms_thresh)[:self.num_detections]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
            results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
            
        return results
    
    def forward(self, features, proposals, batch_shape, image_shapes, targets):
        """
        Args:
            features (List[Tensor[N, C, H, W]]): len(fetures) = PAN levels.
            proposals (List[Tensor[B, 4]])
            batch_shape (Tuple[bH, bW])
            targets (List[Dict])
            
        Returns:
            results (List[Dict])
            losses (Dict): keys: roi_classification_loss, roi_box_loss, roi_mask_loss.
       
        """
        if self.training:
            proposals, matched_ids, labels, regression_targets = self.select_training_samples(proposals, targets)
        
        num_boxes_per_image = [len(b) for b in proposals]
        #print('box roi align boxes', [len(b) for b in proposals])
        box_features = self.box_roi_pool(features, proposals, batch_shape)
        class_logits, box_regression = self.box_predictor(box_features)
        
        results, losses = [], {}
        if self.training:
            classification_loss, box_reg_loss = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = dict(roi_classification_loss=classification_loss, roi_box_loss=box_reg_loss)
        else:
            results = self.fastrcnn_inference(class_logits, box_regression, proposals, image_shapes)
            
        if self.has_mask:
            if self.training:
                num_pos_per_image = [r.shape[0] for r in regression_targets]
                #print('num_pos_per_image', num_pos_per_image)
                mask_proposals = [p[:n] for n, p in zip(num_pos_per_image, proposals)]
                pos_matched_ids = [m[:n] for n, m in zip(num_pos_per_image, matched_ids)]
                mask_labels = [l[:n] for n, l in zip(num_pos_per_image, labels)]
                
                # ---------------------------------------------------------------------
                device = box_regression.device
                box_regression = box_regression.split(num_boxes_per_image)
                for i in range(len(image_shapes)):
                    ids = torch.arange(num_pos_per_image[i], device=device)
                    reg = box_regression[i][ids, mask_labels[i]]
                    mask_proposals[i] = self.box_coder.decode(reg, mask_proposals[i])
                # ---------------------------------------------------------------------
                
            else:
                mask_proposals = [r['boxes'] for r in results]
                
            #print('mask roi align boxes', [len(b) for b in mask_proposals])
            mask_features = self.mask_roi_pool(features, mask_proposals, batch_shape)
            mask_logits = self.mask_predictor(mask_features)
            
            if self.training:
                mask_loss = maskrcnn_loss(mask_logits, mask_proposals, pos_matched_ids, mask_labels, targets)
                losses.update(dict(roi_mask_loss=mask_loss))
            else:
                for i, result in enumerate(results):
                    labels = result['labels']
                    idx = torch.arange(labels.shape[0], device=labels.device)
                    mask_logits = mask_logits[idx, labels]

                    mask_probs = mask_logits.sigmoid()
                    result.update(dict(masks=mask_probs))
                
        return results, losses