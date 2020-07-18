# PyTorch-Simple-MaskRCNN

A PyTorch implementation of simple Mask R-CNN.

This repository is a toy example of Mask R-CNN with two features:
- It is pure python code and can be run immediately using PyTorch 1.4 without build
- Simplified construction and easy to understand how the model works

The code is based largely on [TorchVision](https://github.com/pytorch/vision), but simplified a lot and faster (1.5x).

## Requirements

- Windows with Python ≥ 3.6

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
Note: The code will check the dataset first before start, filtering samples without annotations.

## Training

```
python train.py --use-cuda --iters 200 --dataset coco --data-dir /data/coco2017
```
or modify the parameters in ```run.sh```, and run:
```
bash ./run.sh
```

Note: This is a simple model and only support ```batch_size = 1```. 

The code will save and resume automatically using the checkpoint file.

## Evaluation

- Modify the parameters in ```eval.ipynb``` to test the model.

A good result should be like this:
![example](https://github.com/Okery/PyTorch-Simple-MaskRCNN/blob/master/image/001.png)

## Performance

The model utilizes part of TorchVision's weights which is pretrained on COCO dataset.

Test on VOC 2012 Segmentation val, on 1 RTX 2080Ti:

| model | backbone | imgs/s (train) | imgs/s (test)|epoch | bbox AP | mask AP |
| ---- | ---- | --- | --- | -- | -- | -- |
| Mask R-CNN | ResNet 50 | 11.5 | 15.8 | 5 | 52.2 | 37.0 |
