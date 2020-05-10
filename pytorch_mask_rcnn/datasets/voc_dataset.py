import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict

import torch
import numpy as np
from torchvision import transforms
try:
    import pycocotools.mask as mask_utils
    from pycocotools.coco import COCO
except ImportError:
    pass

from .generalized_dataset import GeneralizedDataset


def target_to_coco_ann(target):
    image_id = target['image_id'].item()
    boxes = target['boxes']
    masks = target['masks']
    labels = target['labels'].tolist()

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
    area = boxes[:, 2] * boxes[:, 3]
    area = area.tolist()
    boxes = boxes.tolist()
    
    rles = [
        mask_utils.encode(np.array(mask[:, :, None], dtype=np.uint8, order='F'))[0]
        for mask in masks
    ]
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    anns = []
    for i, rle in enumerate(rles):
        anns.append(
            {
                'image_id': image_id,
                'id': i,
                'category_id': labels[i],
                'segmentation': rle,
                'bbox': boxes[i],
                'area': area[i],
                'iscrowd': 0,
            }
        )
    return anns     


class VOCDataset(GeneralizedDataset):
    # download VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    def __init__(self, data_dir, split, train=False, max_workers=None):
        self.data_dir = data_dir
        self.split = split
        self.dtype = None
        self._classes = (
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        self.classes = {i: k for i, k in enumerate(self._classes, 1)}
        
        id_file_path = os.path.join(data_dir, 'ImageSets/Segmentation/{}.txt'.format(split))
        ids = [id_.strip() for id_ in open(id_file_path)]
        checked_id_file_path = os.path.join(data_dir, 'ImageSets/Segmentation/checked_{}.txt'.format(split))
        
        super().__init__(ids, train, checked_id_file_path, max_workers)

    def get_image(self, img_id):
        image = Image.open(os.path.join(self.data_dir, 'JPEGImages/{}.jpg'.format(img_id)))
        image = transforms.ToTensor()(image.convert('RGB'))
        return image
        
    def get_target(self, img_id):
        masks = Image.open(os.path.join(self.data_dir, 'SegmentationObject/{}.png'.format(img_id)))
        masks = transforms.ToTensor()(masks)
        uni = masks.unique()
        uni = uni[(uni > 0) & (uni < 1)]
        masks = (masks == uni.reshape(-1, 1, 1)).to(torch.uint8)

        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', '{}.xml'.format(img_id)))
        boxes = []
        labels = []
        for obj in anno.findall('object'):
            bndbox = obj.find('bndbox')
            bbox = [int(bndbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
            name = obj.find('name').text
            label = self._classes.index(name) + 1

            boxes.append(bbox)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)

        img_id = self.ids.index(img_id)
        target = dict(image_id=torch.tensor(img_id), boxes=boxes, labels=labels, masks=masks)
        return target
    
    def convert_to_coco_format(self):
        ann_file = os.path.join(self.data_dir, 'Annotations', 'instances_{}2012.json'.format(self.split))
        if os.path.exists(ann_file):
            self.coco = COCO(ann_file)
            return

        voc_dataset = VOCDataset(self.data_dir, self.split, True)
        instances = defaultdict(list)
        instances['categories'] = [{'id': k, 'name': v} for k, v in voc_dataset.classes.items()]

        ann_id_start = 0
        for image, target in voc_dataset:
            image_id = target['image_id'].item()

            filename = voc_dataset.ids[image_id] + '.jpg'
            h, w = image.shape[-2:]
            img = {'id': image_id, 'file_name': filename, 'height': h, 'width': w}
            instances['images'].append(img)

            anns = target_to_coco_ann(target)
            for ann in anns:
                ann['id'] += ann_id_start
                instances['annotations'].append(ann)
            ann_id_start += len(anns)

        json.dump(instances, open(ann_file, 'w'))
        print('Created COCO-style annotations successfully: {}'.format(ann_file))
        self.coco = COCO(ann_file)