import math
import sys
import torch
try:
    from torch.cuda.amp import autocast
except ImportError:
    pass

from . import distributed
from . import utils
try:
    from .datasets import CocoEvaluator
except ImportError:
    pass


amp = False
if torch.__version__ >= '1.6.0':
    capability = torch.cuda.get_device_capability()[0]
    if capability >= 7:
        amp = True
    

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, iters=None):
    model.train()
    
    iters = len(data_loader) if iters is None else iters
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 0.001
        warmup_iters = min(1000, len(data_loader))

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    if amp:
        print('AMP is enabled for training!')
        
    for i, (images, targets) in enumerate(data_loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]

        if amp:
            with autocast():
                losses = model(images, targets)
        else:
            losses = model(images, targets)
        total_loss = sum(losses.values())
        
        losses_reduced = distributed.reduce_dict(losses)
        total_loss_reduced = sum(losses_reduced.values())
        loss_value = total_loss_reduced.item()
        
        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            print('{}\t'.format(i), '\t'.join('{:.3f}'.format(l.item()) for l in losses_reduced.values()))

        if lr_scheduler is not None:
            lr_scheduler.step()
            
        if i >= iters:
            break
            

@torch.no_grad()
def evaluate(model, data_loader, device, iters=None):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    
    iters = len(data_loader) if iters is None else iters
    dataset = data_loader.dataset
    if not hasattr(dataset, 'coco'):
        dataset.convert_to_coco_format()
    coco = dataset.coco
    iou_types = ['bbox', 'segm']
    coco_evaluator = CocoEvaluator(coco, iou_types)

    if amp:
        print('AMP is enabled for evaluation!')
        
    for i, (images, targets) in enumerate(data_loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]

        torch.cuda.synchronize()
        if amp:
            with autocast():
                results = model(images)
        else:
            results = model(images)

        results = [{k: v.cpu() for k, v in res.items()} for res in results]
        predictions = {tgt['image_id'].item(): res for tgt, res in zip(targets, results)}
        coco_evaluator.update(predictions)
        
        if i >= iters:
            break
            
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    # collect printed information
    temp = sys.stdout
    sys.stdout = utils.TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
    
    torch.set_num_threads(n_threads)

    return output