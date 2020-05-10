import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import MultiScaleRoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer
from .backbone import resnet_pan_backbone
from .predictors import FastRCNNPredictor, MaskRCNNPredictor


class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor], containing the classification and regression losses 
    for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format, 
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
        
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).
        
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the 
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.
        
    """
    
    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.5, box_nms_thresh=0.5, box_num_detections=100):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        
        #------------- RPN --------------------------
        anchor_sizes = (32, 64, 128, 256)
        aspect_ratios = (0.5, 1, 2)
        num_anchors = len(aspect_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        #------------ RoIHeads --------------------------
        box_roi_pool = MultiScaleRoIAlign(output_size=(7, 7), sampling_ratio=2)
        
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)
        
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)
        
        self.head.mask_roi_pool = MultiScaleRoIAlign(output_size=(14, 14), sampling_ratio=2)
        
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        position = 4
        mask_resolution = self.head.mask_roi_pool.output_size[0]
        self.head.mask_predictor = MaskRCNNPredictor(
            out_channels, layers, dim_reduced, num_classes,
            position, mask_resolution)
        
        #------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=800, max_size=1333, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])
        
    def forward(self, images, targets=None):
        orig_image_shapes = [tuple(img.shape[1:]) for img in images]
        #print('orig_image_shapes', orig_image_shapes)
        images, targets, image_shapes = self.transformer(images, targets)
        batch_shape = images.shape[2:]
        features = self.backbone(images)
        
        proposals, rpn_losses = self.rpn(features, batch_shape, image_shapes, targets)
        #s = [len(b) for b in proposals]
        #print('rpn proposals', s)
        results, roi_losses = self.head(features, proposals, batch_shape, image_shapes, targets)
        
        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            results = self.transformer.postprocess(results, image_shapes, orig_image_shapes)
            return results
        
                
def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=False, ckpt_dir=None):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    
    if pretrained:
        backbone_pretrained = False
        
    
    backbone = resnet_pan_backbone('resnet50', pretrained_backbone, ckpt_dir)
    model = MaskRCNN(backbone, num_classes)
    
    if pretrained:
        if ckpt_dir is None:
            model_urls = {
                'maskrcnn_resnet50_fpn_coco':
                    'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
            }
            model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])
        else:
            weights = {
                'maskrcnn_resnet50_fpn_coco': 'maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
            }
            model_state_dict = torch.load(os.path.join(ckpt_dir, weights['maskrcnn_resnet50_fpn_coco']))

        pretrained_msd = list(model_state_dict.values())
        # delete those weights that aren't going to be used
        del_list = [i for i in range(291, 295)] + [305, 306]
        if num_classes == 91:
            del_list = []
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [i for i in range(281, 287)] + [297, 298, 299, 300] + [i for i in range(311, 319)]
        if num_classes == 91:
            skip_list = [i for i in range(281, 287)] + [i for i in range(313, 319)]

        offset = 0
        for i, name in enumerate(msd):
            if i in skip_list:
                offset += 1
                continue
            msd[name].copy_(pretrained_msd[i - offset])

        model.load_state_dict(msd)
    
    return model