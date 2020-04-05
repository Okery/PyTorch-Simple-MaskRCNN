import os
import time
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torchvision import transforms
try:
    from pycocotools.coco import COCO
except ImportError:
    pass

VOC_BBOX_LABEL_NAMES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

        
class VOCDataset:
    # download VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    def __init__(self, data_dir, split, train=False, device='cpu'):
        id_file_path = os.path.join(data_dir, 'ImageSets/Segmentation/{}.txt'.format(split))
        self.ids = [id_.strip() for id_ in open(id_file_path)]
        
        self.data_dir = data_dir
        self.train = train
        self.device = device
        
        if self.train:
            self.check_dataset(split)

    def __getitem__(self, i):
        img_name = self.ids[i]
        image = Image.open(os.path.join(self.data_dir, 'JPEGImages/{}.jpg'.format(img_name)))
        image = transforms.ToTensor()(image)
        image = image.to(self.device)
        
        if self.train:
            masks = Image.open(os.path.join(self.data_dir, 'SegmentationObject/{}.png'.format(img_name)))
            masks = transforms.ToTensor()(masks)
            uni = masks.unique()
            uni = uni[(uni > 0) & (uni < 1)]
            masks = (masks == uni.reshape(-1, 1, 1)).to(torch.uint8)
            masks = masks.to(self.device)
        
            anno = ET.parse(os.path.join(self.data_dir, 'Annotations', '{}.xml'.format(img_name)))
            boxes = []
            labels = []
            for obj in anno.findall('object'):
                bndbox = obj.find('bndbox')
                bbox = [int(bndbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
                name = obj.find('name').text
                label = VOC_BBOX_LABEL_NAMES.index(name)
                
                boxes.append(bbox)
                labels.append(label)

            boxes = torch.tensor(boxes, dtype=image.dtype, device=self.device)
            labels = torch.tensor(labels, device=self.device)
            
            target = dict(boxes=boxes, labels=labels, masks=masks)
            return image, target

        return image
    
    def __len__(self):
        return len(self.ids)
    
    def check_dataset(self, split):
        checked_id_file_path = os.path.join(self.data_dir, 'ImageSets/Segmentation/checked_{}.txt'.format(split))
        if os.path.exists(checked_id_file_path):
            self.ids = [id_.strip() for id_ in open(checked_id_file_path)]
            return
        
        since = time.time()
        print('checking the dataset...')
        with open(checked_id_file_path, 'w') as f:
            for i in range(len(self)):
                img_name = self.ids[i]
                image, target = self[i]
                box = target['boxes']
                mask = target['masks']
                label = target['labels']

                try:
                    assert image.shape[0] == 3, \
                    '{}: image channel != 3, {}'.format(i, image.shape[0])
                    
                    n = torch.where((box[:, 0] < 0) |
                                    (box[:, 1] < 0) |
                                    (box[:, 2] > image.shape[-1]) |
                                    (box[:, 3] > image.shape[-2]))[0]
                    assert len(n) == 0, \
                    '{} box out of boundary'.format(i)
                    
                    assert box.shape[0] == mask.shape[0], \
                    '{}: box not match mask, {}-{}'.format(i, box.shape[0], mask.shape[0])
                    
                    assert box.shape[0] == label.shape[0], \
                    '{}: box not match label, {}-{}'.format(i, box.shape[0], label.shape[0])
                    
                    assert image.shape[-2:] == mask.shape[-2:], \
                    '{}: mask size not match image size, {}-{}'.format(i, image.shape[-2:], mask.shape[-2:])
                    
                    f.write('{}\n'.format(img_name))
                except AssertionError as e:
                    print(img_name, e)

        print('{} check over! {} samples are OK; {:.1f} s'.format(split, len(self), time.time() - since))
        self.ids = [id_.strip() for id_ in open(checked_id_file_path)]
        
        
class COCODataset:
    def __init__(self, data_dir, split, train=False, device='cpu'):
        ann_file = os.path.join(data_dir, 'annotations/instances_{}2017.json'.format(split))
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs)
        
        self.data_dir = data_dir
        self.split = split
        self.train = train
        self.device = device
        
    @staticmethod
    def _convert_box_format(box):
        new_box = torch.zeros_like(box)
        new_box[:, 0] = box[:, 0]
        new_box[:, 1] = box[:, 1]
        new_box[:, 2] = box[:, 0] + box[:, 2]
        new_box[:, 3] = box[:, 1] + box[:, 3]
        return new_box

    def __getitem__(self, i):
        img_id = self.img_ids[i]
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, '{}2017'.format(self.split), img_info['file_name']))
        image = transforms.ToTensor()(image)
        image = image.to(self.device)
        
        if self.train:
            ann_ids = self.coco.getAnnIds(img_id)
            anns = self.coco.loadAnns(ann_ids)
            boxes = []
            labels = []
            masks = []
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann['category_id'])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8, device=self.device)
                masks.append(mask)
            
            boxes = torch.tensor(boxes, dtype=image.dtype, device=self.device)
            boxes = self._convert_box_format(boxes)
            labels = torch.tensor(labels, device=self.device)
            masks = torch.stack(masks)
            
            target = dict(boxes=boxes, labels=labels, masks=masks)
            return image, target

        return image
    
    def __len__(self):
        return len(self.img_ids)
    