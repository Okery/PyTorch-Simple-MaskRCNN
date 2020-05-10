import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
    

def show(image, target=None, classes=None, base=0.4):
    image = image.clone()
        
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
                        s = round(s.item() * 100, 1)
                        txt = '{} {}'.format(txt, s)
                    plt.text(
                        b[0], b[1], txt, fontsize=10, 
                        horizontalalignment='left', verticalalignment='bottom',
                        bbox=dict(boxstyle='round', fc='white', lw=1, alpha=0.8)
                    )
            
    H, W = image.shape[-2:]
    plt.title('H: {}   W: {}'.format(H, W))
    plt.axis('off')
    plt.show()