# PyTorch-Simple-MaskRCNN
A PyTorch implementation of simple Mask R-CNN.

This repository is a toy example of Mask R-CNN with two features:
- It is pure python code and can be run immediately using PyTorch 1.4 without build
- Simplified construction and easy to understand how the model works

The code is based largely on [TorchVision](https://github.com/pytorch/vision)
## Requirements

- python>=3.7

- PyTorch>=1.4

- matplotlib, for visualization

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

Note: I haven't trained the model properly due to the small dataset (2913 train samples), so I don't know whether the model would output good results after trained.

A good result should be like this:
![example](https://github.com/Okery/PyTorch-Simple-MaskRCNN/blob/master/image/001.png)

