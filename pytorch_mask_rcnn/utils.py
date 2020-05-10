import os
import re
import torch
from .datasets import VOCDataset, COCODataset

__all__ = ['gpu_info', 'collate_wrapper', 'datasets', 'save_ckpt', 'parse_eval_output']


def gpu_info(show=False):
    ngpus = torch.cuda.device_count()
    
    properties = []
    for dev in range(ngpus):
        prop = torch.cuda.get_device_properties(dev)
        properties.append({
            'name': prop.name,
            'capability': (prop.major, prop.minor),
            'total_momory': round(prop.total_memory / 1073741824, 2), # unit GB
            'sm_count': prop.multi_processor_count
        })
       
    if show:
        print('cuda: {}'.format(torch.cuda.is_available()))
        print('available GPU(s): {}'.format(ngpus))
        for i, p in enumerate(properties):
            print('{}: {}'.format(i, p))
    return properties


def collate_wrapper(batch):
    return tuple(zip(*batch))


def datasets(ds, *args, **kwargs):
    ds = ds.lower()
    choice = ['voc', 'coco']
    if ds not in choice:
        raise ValueError("'ds' must be in '{}', but got '{}'".format(choice, ds))
    if ds == choice[0]:
        return VOCDataset(*args, **kwargs)
    if ds == choice[1]:
        return COCODataset(*args, **kwargs)
    
    
def save_ckpt(model, optimizer, ckpt_path, lr_scheduler=None, temp=False, **kwargs):
    checkpoint = {}
    checkpoint['model'] = model.state_dict()
    checkpoint['optimizer']  = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint['lr_scheduler']  = lr_scheduler.state_dict()
        
    suffix = []
    for k, v in kwargs.items():
        checkpoint[k] = v
        suffix.append(str(v))
        
    if temp:
        prefix, ext = os.path.splitext(ckpt_path)
        ckpt_path = '-'.join([prefix, *suffix]) + ext
    torch.save(checkpoint, ckpt_path)
    
    
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    
class TextArea:
    def __init__(self):
        self.buffer = []
    
    def write(self, s):
        self.buffer.append(s)
        
    def __str__(self):
        return ''.join(self.buffer)
    
    
def parse_eval_output(txt):
    values = re.findall(r'(\d{3})\n', txt)
    values = [int(v) / 10 for v in values]
    result = {'bbox AP': values[0], 'mask AP': values[12]}
    return result
    