import os
import time
import torch
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed


class GeneralizedDataset:
    """
    Main class for Generalized Dataset.

    Arguments:
        ids (List[str]): images' ids
        train (bool)
        checked_id_file_path (str): path to save the file filled with checked ids.
    """
    
    def __init__(self, ids, train, checked_id_file_path, max_workers):
        self.ids = ids
        self.train = train
        if max_workers is None:
            max_workers = cpu_count() // 2
        self.max_workers = max_workers
        
        if train:
            self.check_dataset(checked_id_file_path)
        
    def __getitem__(self, i):
        """
        Returns:
            image (Tensor): the original image.
            target (Dict[Tensor]): annotations like `boxes`, `labels` and `masks`.
                the `boxes` coordinates order is: xmin, ymin, xmax, ymax
        """
    
        img_id = self.ids[i]
        image = self.get_image(img_id)
        
        if self.train:
            target = self.get_target(img_id)
            return image, target

        return image
        
    def __len__(self):
        return len(self.ids)
    
    def check_dataset(self, checked_id_file_path):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems listed in function `_check`.
        """
        
        if os.path.exists(checked_id_file_path):
            self.ids = [id_.strip() for id_ in open(checked_id_file_path)]
            return
        
        print('Checking the dataset...')
        
        since = time.time()
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        tasks = []
        
        file_paths = []
        seqs = torch.arange(len(self)).chunk(self.max_workers)
        for i, seq in enumerate(seqs, 1):
            if i == 1:
                file_path = checked_id_file_path
            else:
                file_dir, file_name = os.path.split(checked_id_file_path)
                file_name = '{}.{}'.format(i, file_name)
                file_path = os.path.join(file_dir, file_name)
                file_paths.append(file_path)
                
            tasks.append(executor.submit(self._check, file_path, seq.tolist()))

        for future in as_completed(tasks):
            pass
        
        with open(checked_id_file_path, 'a') as f:
            for p in file_paths:
                with open(p) as f1:
                    f.write(f1.read())
                os.remove(p)
                
        self.ids = [id_.strip() for id_ in open(checked_id_file_path)]
        print('{} check over! {} samples are OK; {:.1f} s'.format(checked_id_file_path, len(self), time.time() - since))
        
    def _check(self, path, seq):
        with open(path, 'w') as f:
            for i in seq:
                img_id = self.ids[i]
                image, target = self[i]
                box = target['boxes']
                mask = target['masks']

                try:
                    assert len(box) > 0, '{}: len(box) = 0'.format(i)

                    assert len(box) == len(mask), \
                    '{}: box not match mask, {}-{}'.format(i, box.shape[0], mask.shape[0])

                    f.write('{}\n'.format(img_id))
                except AssertionError as e:
                    print(img_id, e)