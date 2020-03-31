import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from . import dataset


classes = dataset.VOC_BBOX_LABEL_NAMES

def factor_getter(i):
    base = 0.5
    f = [0, 0, 0]
    if i < 3:
        f[i] = base
    elif i < 6:
        base /= 2
        f = [base, base, base]
        f[i - 3] = 0
    else:
        base /= 3
        f = [base, base, base]
    return f


def resize(image, target, scale_factor):
    ori_image_shape = image.shape[-2:]
    image = F.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

    if target is None:
        return image, target

    if 'boxes' in target:
        box = target['boxes']
        box[:, [0, 2]] = box[:, [0, 2]] * image.shape[-1] / ori_image_shape[1]
        box[:, [1, 3]] = box[:, [1, 3]] * image.shape[-2] / ori_image_shape[0]
        target['boxes'] = box

    if 'masks' in target:
        mask = target['masks']
        mask = F.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
        target['masks'] = mask

    return image, target
    

def show(image, target=None, scale_factor=None):
    image = image.clone()
    
    if scale_factor is not None:
        image, target = resize(image, target, scale_factor)
        
    if target is not None and 'masks' in target:
        mask = target['masks']
        mask = mask.reshape(-1, 1, mask.shape[1], mask.shape[2])
        mask = mask.repeat(1, 3, 1, 1).to(image)
        for i, m in enumerate(mask):
            factor = torch.tensor(factor_getter(i)).reshape(3, 1, 1).to(image)
            value = factor * m
            image += value
        
    image = image.clamp(0, 1)
    im = image.cpu().numpy()
    plt.imshow(im.transpose(1, 2, 0))
    
    if target is not None:
        if 'boxes' in target:
            box = target['boxes']
            box = box.cpu()
            for i, b in enumerate(box):
                plt.plot(b[[0, 2, 2, 0, 0]], b[[1, 1, 3, 3, 1]])
                if 'labels' in target:
                    l = target['labels'][i]
                    txt = classes[l]
                    if 'scores' in target:
                        s = target['scores'][i]
                        s = round(s.item() * 100)
                        txt = '{} {}%'.format(txt, s)
                    plt.text(
                        b[0], b[1], txt, fontsize=15, 
                        bbox=dict(boxstyle='round, pad=0.2', fc='white', lw=1, alpha=0.5))# , ec='k'  
            
    plt.title(im.shape)
    plt.axis('off')
    plt.show()


def generate_bbox(num, size, manual_seed=True):
    if manual_seed:
        torch.manual_seed(3)
    x = torch.randint(0, size[1], (2, num), dtype=torch.float32)
    y = torch.randint(0, size[0], (2, num), dtype=torch.float32)
    
    ymin = y.min(dim=0)[0]
    xmin = x.min(dim=0)[0]
    ymax = y.max(dim=0)[0] + 1
    xmax = x.max(dim=0)[0] + 1
    
    bbox = torch.stack((xmin, ymin, xmax, ymax), dim=1)
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