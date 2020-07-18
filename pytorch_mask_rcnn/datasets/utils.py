import torch

from .voc_dataset import VOCDataset
from .coco_dataset import COCODataset

__all__ = ["datasets", "collate_wrapper", "DataPrefetcher"]


def datasets(ds, *args, **kwargs):
    ds = ds.lower()
    choice = ["voc", "coco"]
    if ds == choice[0]:
        return VOCDataset(*args, **kwargs)
    if ds == choice[1]:
        return COCODataset(*args, **kwargs)
    else:
        raise ValueError("'ds' must be in '{}', but got '{}'".format(choice, ds))
    
    
def collate_wrapper(batch):
    return CustomBatch(batch)

    
class CustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.images = transposed_data[0]
        self.targets = transposed_data[1]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.images = [img.pin_memory() for img in self.images]
        self.targets = [{k: v.pin_memory() for k, v in tgt.items()} for tgt in self.targets]
        return self
    

class DataPrefetcher:
    def __init__(self, data_loader):
        self.loader = data_loader
        self.dataset = data_loader.dataset
        self.stream = torch.cuda.Stream()
        
    def __iter__(self):
        for i, d in enumerate(self.loader, 1):
            if i == 1:
                d.images = [img.cuda(non_blocking=True) for img in d.images]
                d.targets = [{k: v.cuda(non_blocking=True) for k, v in tgt.items()} for tgt in d.targets]
                self._cache = d
                continue
               
            torch.cuda.current_stream().wait_stream(self.stream)
            out = self._cache
            
            with torch.cuda.stream(self.stream):
                d.images = [img.cuda(non_blocking=True) for img in d.images]
                d.targets = [{k: v.cuda(non_blocking=True) for k, v in tgt.items()} for tgt in d.targets]
            self._cache = d
            yield out
        yield self._cache
        
    def __len__(self):
        return len(self.loader)
    
