import os
import time
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torchvision import transforms

VOC_BBOX_LABEL_NAMES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

        
class VOCDataset:
    # download VOC 2012: 
    # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    def __init__(self, data_dir, split, train, device='cpu'):
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
            mask = Image.open(os.path.join(self.data_dir, 'SegmentationObject/{}.png'.format(img_name)))
            mask = transforms.ToTensor()(mask)
            uni = mask.unique()
            uni = uni[(uni > 0) & (uni < 1)]
            mask = (mask == uni.reshape(-1, 1, 1)).to(torch.uint8)
            mask = mask.to(self.device)
        
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
            
            target = dict(boxes=boxes, labels=labels, masks=mask)
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