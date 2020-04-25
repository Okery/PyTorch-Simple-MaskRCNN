import copy
import torch
import numpy as np
import pycocotools.mask as mask_utils

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {k: COCOeval(coco_gt, iouType=k) for k in iou_types}

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(predictions.keys())
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_dt = self.coco_gt.loadRes(results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = img_ids
            coco_eval.evaluate()

            self.eval_imgs[iou_type].extend(coco_eval.evalImgs)
            
    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            coco_eval = self.coco_eval[iou_type]
            coco_eval.evalImgs = self.eval_imgs[iou_type]
            coco_eval.params.imgIds = self.img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            
    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print('IoU metric: {}'.format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == 'bbox':
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == 'segm':
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError('Unknown iou type {}'.format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for image_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            # convert to coco format: x, y, w, h
            boxes = prediction['boxes']
            xmin, ymin, xmax, ymax = boxes.unbind(1)
            boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
            boxes = boxes.tolist()
            
            scores = prediction['scores'].tolist()
            labels = prediction['labels'].tolist()

            coco_results.extend(
                [
                    {
                        'image_id': image_id,
                        'category_id': labels[k],
                        'bbox': box,
                        'score': scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for image_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction['scores']
            labels = prediction['labels']
            masks = prediction['masks']

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_utils.encode(np.array(mask[:, :, None], dtype=np.uint8, order='F'))[0]
                for mask in masks
            ]
            for rle in rles:
                rle['counts'] = rle['counts'].decode('utf-8')

            coco_results.extend(
                [
                    {
                        'image_id': image_id,
                        'category_id': labels[k],
                        'segmentation': rle,
                        'score': scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results