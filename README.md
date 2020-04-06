# PyTorch-Simple-MaskRCNN

A PyTorch implementation of simple Mask R-CNN.

This repository is a toy example of Mask R-CNN with two features:
- It is pure python code and can be run immediately using PyTorch 1.4 without build
- Simplified construction and easy to understand how the model works

The code is based largely on [TorchVision](https://github.com/pytorch/vision), but simplified a lot and faster (1.5x).

## Requirements

- Windows with Python ≥ 3.6

- [PyTorch](https://pytorch.org/) ≥ 1.4

- Torchvision that matches the PyTorch installation

optional:

- matplotlib, needed by visualization

- pycocotools, needed by COCO dataset

## Datasets

[PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (click to download).
```
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
```
MS COCO 2017
```
http://cocodataset.org/
```
Note: The code will check the dataset first before start. It is very necessary and may take much time if the dataset is large. Do not stop it for it's just once.

## Training

- Adjust parameters in ```train.ipynb``` to train the model

Note: This is a simple model and only support ```batch_size = 1```. Set ```epochs = n``` to train n epochs, the model will save and resume automatically using the ```checkpoint.pth``` file.

## Evaluation

- Adjust parameters in ```eval.ipynb``` to test the model

A good result should be like this:
![example](https://github.com/Okery/PyTorch-Simple-MaskRCNN/blob/master/image/001.png)

## Performance

The model is pretrained on COCO dataset.

Test on VOC 2012 Segmentation val:

bbox:

| model | backbone | epoch | mAP | AP50 | AP75 |
| ---- | ---- | --- | -- | -- | -- |
| Mask R-CNN | ResNet 50 | 15 | 59.7 | 88.0 | 68.3 |
