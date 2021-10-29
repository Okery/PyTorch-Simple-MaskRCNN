import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict

import torch
import numpy as np
import pycocotools.mask as mask_util
from torchvision import transforms

from .generalized_dataset import GeneralizedDataset


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
)

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
        mask_util.encode(np.array(mask[:, :, None], dtype=np.uint8, order='F'))[0]
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
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        # instances segmentation task
        id_file = os.path.join(data_dir, "ImageSets/Segmentation/{}.txt".format(split))
        self.ids = [id_.strip() for id_ in open(id_file)]
        self.id_compare_fn = lambda x: int(x.replace("_", ""))
        
        self.ann_file = os.path.join(data_dir, "Annotations/instances_{}.json".format(split))
        self._coco = None
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = {i: n for i, n in enumerate(VOC_CLASSES, 1)}
        
        checked_id_file = os.path.join(os.path.dirname(id_file), "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self.make_aspect_ratios()
            self.check_dataset(checked_id_file)
            
    def make_aspect_ratios(self):
        self._aspect_ratios = []
        for img_id in self.ids:
            anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id)))
            size = anno.findall("size")[0]
            width = size.find("width").text
            height = size.find("height").text
            ar = int(width) / int(height)
            self._aspect_ratios.append(ar)

    def get_image(self, img_id):
        image = Image.open(os.path.join(self.data_dir, "JPEGImages/{}.jpg".format(img_id)))
        return image.convert("RGB")
        
    def get_target(self, img_id):
        masks = Image.open(os.path.join(self.data_dir, 'SegmentationObject/{}.png'.format(img_id)))
        masks = transforms.ToTensor()(masks)
        uni = masks.unique()
        uni = uni[(uni > 0) & (uni < 1)]
        masks = (masks == uni.reshape(-1, 1, 1)).to(torch.uint8)
        
        anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id)))
        boxes = []
        labels = []
        for obj in anno.findall("object"):
            bndbox = obj.find("bndbox")
            bbox = [int(bndbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
            name = obj.find("name").text
            label = VOC_CLASSES.index(name) + 1

            boxes.append(bbox)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)

        img_id = torch.tensor([self.ids.index(img_id)])
        target = dict(image_id=img_id, boxes=boxes, labels=labels, masks=masks)
        return target
    
    @property
    def coco(self):
        if self._coco is None:
            from pycocotools.coco import COCO
            self.convert_to_coco_format()
            self._coco = COCO(self.ann_file)
        return self._coco
    
    def convert_to_coco_format(self, overwrite=False):
        if overwrite or not os.path.exists(self.ann_file):
            print("Generating COCO-style annotations...")
            voc_dataset = VOCDataset(self.data_dir, self.split, True)
            instances = defaultdict(list)
            instances["categories"] = [{"id": i + 1, "name": n} for i, n in enumerate(VOC_CLASSES)]

            ann_id_start = 0
            for image, target in voc_dataset:
                image_id = target["image_id"].item()

                filename = voc_dataset.ids[image_id] + ".jpg"
                h, w = image.shape[-2:]
                img = {"id": image_id, "file_name": filename, "height": h, "width": w}
                instances["images"].append(img)

                anns = target_to_coco_ann(target)
                for ann in anns:
                    ann["id"] += ann_id_start
                    instances["annotations"].append(ann)
                ann_id_start += len(anns)

            json.dump(instances, open(self.ann_file, "w"))
            print("Created successfully: {}".format(self.ann_file))
        
  