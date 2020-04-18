import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

import torch
from torchvision import transforms
try:
    from pycocotools.coco import COCO
except ImportError:
    pass


class GeneralizedDataset:
    """
    Main class for Generalized Dataset.

    Arguments:
        ids (list[str]): images' ids
        train (bool)
        checked_id_file_path (str): path to save the file filled with checked ids.
    """
    
    def __init__(self, ids, train, checked_id_file_path):
        self.ids = ids
        self.train = train
        self.max_workers = 4
        
        if train:
            self.check_dataset(checked_id_file_path)
        
    def __getitem__(self, i):
        """
        Returns:
            image (Tensor): the original image.
            target (dict[Tensor]): annotations like `boxes`, `labels` and `masks`.
                the `boxes` coordinates order is: xmin, ymin, xmax, ymax
        """
    
        img_id = self.ids[i]
        image = self.get_image(img_id)
        
        if self.train:
            target = self.get_target(img_id)
            return image, target

        return image
        
    def __len__(self):
        return len(self.ids)
    
    def check_dataset(self, checked_id_file_path):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems in function `_check`.
        """
        if os.path.exists(checked_id_file_path):
            self.ids = [id_.strip() for id_ in open(checked_id_file_path)]
            return
        
        print('checking the dataset...')
        
        since = time.time()
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        tasks = []
        with open(checked_id_file_path, 'w') as f:
            seqs = torch.arange(len(self)).chunk(self.max_workers)
            for seq in seqs:
                tasks.append(executor.submit(self._check, f, seq.tolist()))
                
            for future in as_completed(tasks):
                pass

        self.ids = [id_.strip() for id_ in open(checked_id_file_path)]
        print('{} check over! {} samples are OK; {:.1f} s'.format(checked_id_file_path, len(self), time.time() - since))
        
    def _check(self, f, seq):
        for i in seq:
            img_id = self.ids[i]
            image, target = self[i]
            box = target['boxes']
            mask = target['masks']
            label = target['labels']
            
            try:
                # some problems below:
                assert image.shape[0] == 3, \
                '{}: image channel != 3, {}'.format(i, image.shape[0])

                assert len(box) > 0, \
                '{}: len(box) = 0'.format(i)

                assert len(box) == len(mask), \
                '{}: box not match mask, {}-{}'.format(i, box.shape[0], mask.shape[0])

                assert len(box) == len(label), \
                '{}: box not match label, {}-{}'.format(i, box.shape[0], label.shape[0])

                n = torch.where((box[:, 0] < 0) |
                                (box[:, 1] < 0) |
                                (box[:, 2] > image.shape[-1]) |
                                (box[:, 3] > image.shape[-2]))[0]
                assert len(n) == 0, \
                '{} box out of boundary'.format(i)

                assert image.shape[-2:] == mask.shape[-2:], \
                '{}: mask size not match image size, {}-{}'.format(i, image.shape[-2:], mask.shape[-2:])
                
                f.write('{}\n'.format(img_id))
            except AssertionError as e:
                print(img_id, e)
        
        
class VOCDataset(GeneralizedDataset):
    # download VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    def __init__(self, data_dir, split, train, device):
        self.data_dir = data_dir
        self.device = device
        self.dtype = None
        self._classes = (
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        self.classes = {i:self._classes[i] for i in range(len(self._classes))}
        
        id_file_path = os.path.join(data_dir, 'ImageSets/Segmentation/{}.txt'.format(split))
        ids = [id_.strip() for id_ in open(id_file_path)]
        checked_id_file_path = os.path.join(data_dir, 'ImageSets/Segmentation/checked_{}.txt'.format(split))
        
        super().__init__(ids, train, checked_id_file_path)

    def get_image(self, img_id):
        image = Image.open(os.path.join(self.data_dir, 'JPEGImages/{}.jpg'.format(img_id)))
        image = transforms.ToTensor()(image)
        image = image.to(self.device)
        return image
        
    def get_target(self, img_id):
        masks = Image.open(os.path.join(self.data_dir, 'SegmentationObject/{}.png'.format(img_id)))
        masks = transforms.ToTensor()(masks)
        uni = masks.unique()
        uni = uni[(uni > 0) & (uni < 1)]
        masks = (masks == uni.reshape(-1, 1, 1)).to(torch.uint8)
        masks = masks.to(self.device)

        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', '{}.xml'.format(img_id)))
        boxes = []
        labels = []
        for obj in anno.findall('object'):
            bndbox = obj.find('bndbox')
            bbox = [int(bndbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
            name = obj.find('name').text
            label = self._classes.index(name)

            boxes.append(bbox)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float, device=self.device)
        labels = torch.tensor(labels, device=self.device)

        target = dict(boxes=boxes, labels=labels, masks=masks)
        return target
        
        
class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train, device):
        ann_file = os.path.join(data_dir, 'annotations/instances_{}2017.json'.format(split))
        self.coco = COCO(ann_file)
        self.data_dir = data_dir
        self.split = split
        self.device = device
        self.classes = {(k - 1):v['name'] for k, v in self.coco.cats.items()}
        
        ids = list(self.coco.imgs)
        checked_id_file_path = os.path.join(data_dir, 'checked_{}.txt'.format(split))
        
        super().__init__(ids, train, checked_id_file_path)
        
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
        image = Image.open(os.path.join(self.data_dir, '{}2017'.format(self.split), img_info['file_name']))
        image = transforms.ToTensor()(image)
        image = image.to(self.device)
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
                labels.append(ann['category_id'] - 1)
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8, device=self.device)
                masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float, device=self.device)
            boxes = self._convert_box_format(boxes)
            labels = torch.tensor(labels, device=self.device)
            masks = torch.stack(masks)

        target = dict(boxes=boxes, labels=labels, masks=masks)
        return target
    

def datasets(d, data_dir, split, train=False, device='cpu'):
    d = d.lower()
    choice = ['voc', 'coco']
    if d not in choice:
        raise ValueError("'d' must be in '{}', but got '{}'".format(choice, d))
    if d == choice[0]:
        return VOCDataset(data_dir, split, train, device)
    if d == choice[1]:
        return COCODataset(data_dir, split, train, device)