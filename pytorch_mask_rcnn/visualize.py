import torch
import torch.nn.functional as F
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def factor_getter(n, base):
    base = base * 0.8 ** (n // 6)
    i = n % 6
    if i < 3:
        f = [0, 0, 0]
        f[i] = base
    else:
        base /= 2
        f = [base, base, base]
        f[i - 3] = 0
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
    

def show(image, target=None, classes=None, scale_factor=None, base=0.4):
    image = image.clone()
    
    if scale_factor is not None:
        image, target = resize(image, target, scale_factor)
        
    if target is not None and 'masks' in target:
        mask = target['masks']
        mask = mask.reshape(-1, 1, mask.shape[1], mask.shape[2])
        mask = mask.repeat(1, 3, 1, 1).to(image)
        for i, m in enumerate(mask):
            factor = torch.tensor(factor_getter(i, base)).reshape(3, 1, 1).to(image)
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
                    l = target['labels'][i].item()
                    if classes is None:
                        raise ValueError("'classes' should not be None when 'target' has 'labels'!")
                    txt = classes[l]
                    if 'scores' in target:
                        s = target['scores'][i]
                        s = round(s.item() * 100)
                        txt = '{} {}%'.format(txt, s)
                    plt.text(
                        b[0], b[1], txt, fontsize=14, 
                        bbox=dict(boxstyle='round', fc='white', lw=1, alpha=0.7))
            
    plt.title(im.shape)
    plt.axis('off')
    plt.show()