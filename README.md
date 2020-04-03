# PyTorch-Simple-MaskRCNN

A PyTorch implementation of simple Mask R-CNN.

This repository is a toy example of Mask R-CNN with two features:
- It is pure python code and can be run immediately using PyTorch 1.4 without build
- Simplified construction and easy to understand how the model works

The code is based largely on [TorchVision](https://github.com/pytorch/vision), but simplified a lot and faster (1.5x).

## Requirements

- Windows with Python ≥ 3.7

- PyTorch ≥ 1.4, 

- torchvision that matches the PyTorch installation

- matplotlib, needed by visualization

## Datasets

Currently only support VOC 2012
```
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
```


## Train

- Adjust parameters in ```train.ipynb``` to train the model

Note: This is a simple model and only support ```batch_size = 1```. Set ```epochs = n``` to train n epochs, the model will save and resume automatically using the ```checkpoint.pth``` file.

## Evaluation

- Adjust parameters in ```eval.ipynb``` to test the model

Note: I haven't trained the model properly due to the small dataset (1463 train samples), so I'm not sure whether the model would output good results after trained.

A good result should be like this:
![example](https://github.com/Okery/PyTorch-Simple-MaskRCNN/blob/master/image/001.png)

##Performance

VOC2012 Segmentation val

bbox:
| model | backbone | epoch | mAP | AP50 | AP 75 |
| Mask R-CNN | ResNet 50 | 15 | 59.7 | 88.0 | 68.3 |
