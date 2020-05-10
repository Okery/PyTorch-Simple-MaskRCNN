import os
from PIL import Image

import torch
from torchvision import transforms
try:
    from pycocotools.coco import COCO
except:
    pass

from .generalized_dataset import GeneralizedDataset
       
        
class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train=False, max_workers=None, year='2017'):
        ann_file = os.path.join(data_dir, 'annotations/instances_{}{}.json'.format(split, year))
        self.coco = COCO(ann_file)
        self.data_dir = data_dir
        self.split = split
        self.year = year
        self.classes = {k: v['name'] for k, v in self.coco.cats.items()}
        
        ids = list(self.coco.imgs)
        checked_id_file_path = os.path.join(data_dir, 'checked_{}.txt'.format(split))
        
        super().__init__(ids, train, checked_id_file_path, max_workers)
        
    @staticmethod
    def _convert_box_format(box):
        new_box = torch.zeros_like(box)
        new_box[:, 0] = box[:, 0]
        new_box[:, 1] = box[:, 1]
        new_box[:, 2] = box[:, 0] + box[:, 2]
        new_box[:, 3] = box[:, 1] + box[:, 3]
        return new_box
    
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, '{}{}'.format(self.split, self.year), img_info['file_name']))
        image = transforms.ToTensor()(image.convert('RGB'))
        return image
        
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann['category_id'])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self._convert_box_format(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)

        target = dict(image_id=torch.tensor(img_id), boxes=boxes, labels=labels, masks=masks)
        return target