# -*- coding: utf-8 -*-
import os
import os.path
from queue import Queue
from threading import Thread

import torch
from PIL import Image


def get_loader(my_dataset, device, batch_size, num_workers, shuffle):
    """ 根据dataset及设置，获取对应的 DataLoader """
    my_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, num_workers=num_workers,
                                            shuffle=shuffle, pin_memory=True, persistent_workers=True)
    if torch.cuda.is_available():
        my_loader = CudaDataLoader(my_loader, device=device)
    return my_loader


class DatasetFolder(torch.utils.data.Dataset):

    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    def __init__(self, root, transform=None, target_transform=None):
        """ dataset """
        samples = []
        for sub_root, _, filenames in sorted(os.walk(root)):
            for filename in sorted(filenames):
                if os.path.splitext(filename)[-1].lower() in self.IMG_EXTENSIONS:
                    path = os.path.join(sub_root, filename)
                    item = (path, 0)
                    samples.append(item)

        if len(samples) == 0:
            raise RuntimeError(f"Found 0 files in sub-folders of: {root}\n"
                               f"Supported extensions are: {','.join(self.IMG_EXTENSIONS)}")

        self.root = root
        self.loader = self.default_loader
        self.extensions = self.IMG_EXTENSIONS
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = f'Dataset {self.__class__.__name__}\n'\
                  f'    Number of data points: {self.__len__()}\n'\
                  f'    Root Location: {self.root}\n'
        tmp = '    Transforms (if any): '
        trans_tmp = self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        fmt_str += f'{tmp}{trans_tmp}\n'
        tmp = '    Target Transforms (if any): '
        trans_tmp = self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        fmt_str += f'{tmp}{trans_tmp}'
        return fmt_str

    @staticmethod
    def default_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class CudaDataLoader:
    """ 异步预先将数据从CPU加载到GPU中 """

    def __init__(self, loader, device, queue_size=2):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        """ 不断的将cuda数据加载到队列里 """
        # The loop that will load into the queue in the background
        torch.cuda.set_device(self.device)
        while True:
            for i, sample in enumerate(self.loader):
                self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        """ 将batch数据从CPU加载到GPU中 """
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) in (list, str):
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        # 加载线程挂了
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个epoch加载完了
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        # 下一个batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
