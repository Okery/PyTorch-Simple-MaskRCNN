from .voc_dataset import VOCDataset
from .coco_dataset import COCODataset

__all__ = ["datasets", "collate_wrapper"]


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
    
