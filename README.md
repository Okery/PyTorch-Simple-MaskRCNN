# PyTorch-Simple-MaskRCNN
A PyTorch implementation of simple Mask R-CNN.

This repository is a toy example Mask R-CNN with two features:
- It is pure python code and can be run immediately using PyTorch 1.4 without build
- Simplified construction and easy to understand how the model works

## Requirements

python 3.7

pytorch 1.4

matplotlib

## Datasets

Currently only support VOC 2012
```
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
```

## Train

- Adjust parameters in ```train.ipynb``` to train the model

NOTE: This is a simple model and only support ```batch_size = 1```. Set ```train_num_samples = n``` to train n samples, the model will save and resume automatically using the ```checkpoint.pth``` file.

## Evaluation

- Adjust parameters in ```eval.ipynb``` to test the model

one example:
![example](https://github.com/Jaramies/PyTorch-Simple-MaskRCNN/blob/master/image/001.png)
- I haven't trained the model properly due to the device limit, so I don't know whether the model would output good results after trained.

