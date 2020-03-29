import torch
import matplotlib.pyplot as plt
import numpy as np
import time


def factor_getter(i):
    base = 0.5
    f = [0, 0, 0]
    if i < 3:
        f[i] = base
    elif i < 6:
        base /= 2
        f = [base, base, base]
        f[i - 3] = 0
    return f
    

def show(image, box=None, mask=None): # show image, just for test
    image = image.clone()
    if mask is not None:
        mask = mask.reshape(-1, 1, mask.shape[1], mask.shape[2])
        mask = mask.repeat(1, 3, 1, 1).to(image)
        for i, m in enumerate(mask):
            factor = torch.tensor(factor_getter(i)).reshape(3, 1, 1).to(image)
            value = factor * m
            image += value
        
    image = image.clamp(0, 1)
    im = image.cpu().numpy()
    plt.imshow(im.transpose(1, 2, 0))
    
    if box is not None:
        box = box.cpu()
        for b in box:
            plt.plot(b[[1, 1, 3, 3, 1]], b[[0, 2, 2, 0, 0]])
            
    plt.title(im.shape)
    plt.axis('off')
    plt.show()


def generate_bbox(num, size, manual_seed=True):
    if manual_seed:
        torch.manual_seed(3)
    y = torch.randint(0, size[0], (2, num), dtype=torch.float32)
    x = torch.randint(0, size[1], (2, num), dtype=torch.float32)
    
    ymin = y.min(dim=0)[0]
    xmin = x.min(dim=0)[0]
    ymax = y.max(dim=0)[0] + 1
    xmax = x.max(dim=0)[0] + 1
    
    bbox = torch.stack((ymin, xmin, ymax, xmax), dim=1)
    return bbox


class Meter:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:sum={sum:.2f}, avg={avg:.2f}, count={count}'
        return fmtstr.format(**self.__dict__)