# PyTorch-Simple-MaskRCNN

A PyTorch implementation of Mask R-CNN with PANet.

This repository owns two features:
- It is pure python code and can be run immediately using PyTorch 1.4 without build
- Simplified structure and easy to understand how the model works

The code is based largely on [TorchVision](https://github.com/pytorch/vision).

## Requirements

- Python ≥ 3.6

- [PyTorch](https://pytorch.org/) ≥ 1.4.0

- Torchvision that matches the PyTorch installation

optional:

- matplotlib, needed by visualization

- [pycocotools](https://github.com/cocodataset/cocoapi) for COCO dataset and evaluation; Windows version is [here](https://github.com/philferriere/cocoapi)

There is a problem with pycocotools for Windows. See [Issue #356](https://github.com/cocodataset/cocoapi/issues/356).

Besides, it's better to remove the prints in pycocotools.

## Datasets

[PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) (click to download).
```
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
```
MS COCO 2017
```
http://cocodataset.org/
```
Note: The code will check the dataset first before start. It will remove some images without annotations. This step is very necessary and may take much time if the dataset is large. Do not stop it for it's just once.

## Training

- Modify parameters in ```train.py``` to train the model

## Demo

- Modify parameters in ```demo.ipynb``` to test the model

Some results:
![example](https://github.com/Okery/PyTorch-Simple-MaskRCNN/blob/master/image/001.png)

## Performance

The model utlizes the official pre-trained weights of Mask R-CNN model and is finetuned in different datasets.

VOC 2012 Segmentation val:

| model | backbone | epoch | bbox AP | AP50 | AP75 | mask AP | AP50| AP75|
| ---- | ---- | --- | -- | -- | -- | -- | -- | -- |
| Mask R-CNN | ResNet 50 PANet | 21 | 42.3 | 67.5 | 47.2 | 35.3 | 58.7 | 37.8 |
