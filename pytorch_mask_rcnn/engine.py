import math
import sys
import torch

from . import utils
from .datasets import CocoEvaluator


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 0.001
        warmup_iters = min(1000, len(data_loader))

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i, (images, targets) in enumerate(data_loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]

        losses = model(images, targets)
        total_loss = sum(losses.values())
        
        if not math.isfinite(total_loss.item()):
            print('Loss is {}, stopping training'.format(total_loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            print('{}\t'.format(i), '\t'.join(str(round(l.item(), 3)) for l in losses.values()))

        if lr_scheduler is not None:
            lr_scheduler.step()
            

@torch.no_grad()
def evaluate(model, data_loader, device, iters=None):
    model.eval()
    
    iters = len(data_loader) if iters is None else iters
    dataset = data_loader.dataset
    if not hasattr(dataset, 'coco'):
        dataset.convert_to_coco_format()
    coco = dataset.coco
    iou_types = ['bbox', 'segm']
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for i, (images, targets) in enumerate(data_loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]

        results = model(images)

        results = [{k: v.cpu() for k, v in res.items()} for res in results]
        predictions = {tgt['image_id'].item(): res for tgt, res in zip(targets, results)}
        coco_evaluator.update(predictions)
        
        if i >= iters:
            break
        
    coco_evaluator.accumulate()
    # collect printed information
    temp = sys.stdout
    sys.stdout = utils.TextArea()

    coco_evaluator.summarize()

    output = str(sys.stdout)
    sys.stdout = temp
    
    return output