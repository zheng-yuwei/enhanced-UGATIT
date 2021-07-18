# -*- coding: utf-8 -*-
import os
import os.path
from queue import Queue
from threading import Thread

import cv2
import torch
import torch.utils.data
import numpy as np

from histogram import match_histograms


def get_loader(my_dataset, device, batch_size, num_workers, shuffle):
    """ 根据dataset及设置，获取对应的 DataLoader """
    my_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, num_workers=num_workers,
                                            shuffle=shuffle, pin_memory=True, persistent_workers=(num_workers > 0))
    # if torch.cuda.is_available():
    #     my_loader = CudaDataLoader(my_loader, device=device)
    return my_loader


class MatchHistogramsDataset(torch.utils.data.Dataset):

    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    def __init__(self, root, transform=None, target_transform=None, is_match_histograms=False, match_mode=True,
                 b2a_prob=0.5, match_ratio=1.0):
        """ 获取指定的两个文件夹下，两张图像numpy数组的Dataset """
        assert len(root) == 2, f'root of MatchHistogramsDataset must has two dir!'
        self.dataset_0 = DatasetFolder(root[0])
        self.dataset_1 = DatasetFolder(root[1])

        self.transform = transform
        self.target_transform = target_transform
        self.len_0 = len(self.dataset_0)
        self.len_1 = len(self.dataset_1)
        self.len = max(self.len_0, self.len_1)
        self.is_match_histograms = is_match_histograms
        self.match_mode = match_mode
        assert self.match_mode in ('hsv', 'hsl', 'rgb'), f'match mode must in {self.match_mode}'
        self.b2a_prob = b2a_prob
        self.match_ratio = match_ratio

    def __getitem__(self, index):
        sample_0 = self.dataset_0[index] if index < self.len_0 else self.dataset_0[np.random.randint(self.len_0)]
        sample_1 = self.dataset_1[index] if index < self.len_1 else self.dataset_1[np.random.randint(self.len_1)]

        if self.is_match_histograms:
            if self.match_mode == 'hsv':
                sample_0 = cv2.cvtColor(sample_0, cv2.COLOR_RGB2HSV_FULL)
                sample_1 = cv2.cvtColor(sample_1, cv2.COLOR_RGB2HSV_FULL)
            elif self.match_mode == 'hsl':
                sample_0 = cv2.cvtColor(sample_0, cv2.COLOR_RGB2HLS_FULL)
                sample_1 = cv2.cvtColor(sample_1, cv2.COLOR_RGB2HLS_FULL)

            if np.random.rand() < self.b2a_prob:
                sample_1 = match_histograms(sample_1, sample_0, rate=self.match_ratio)
            else:
                sample_0 = match_histograms(sample_0, sample_1, rate=self.match_ratio)

            if self.match_mode == 'hsv':
                sample_0 = cv2.cvtColor(sample_0, cv2.COLOR_HSV2RGB_FULL)
                sample_1 = cv2.cvtColor(sample_1, cv2.COLOR_HSV2RGB_FULL)
            elif self.match_mode == 'hsl':
                sample_0 = cv2.cvtColor(sample_0, cv2.COLOR_HLS2RGB_FULL)
                sample_1 = cv2.cvtColor(sample_1, cv2.COLOR_HLS2RGB_FULL)

        if self.transform is not None:
            sample_0 = self.transform(sample_0)
            sample_1 = self.transform(sample_1)

        return sample_0, sample_1

    def __len__(self):
        return self.len

    def __repr__(self):
        fmt_str = f'MatchHistogramsDataset for: \n' \
                  f'{self.dataset_0.__repr__()} \n ' \
                  f'{self.dataset_1.__repr__()}'
        return fmt_str


class DatasetFolder(torch.utils.data.Dataset):

    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    def __init__(self, root, transform=None):
        """ 获取指定文件夹下，单张图像numpy数组的Dataset """
        samples = []
        for sub_root, _, filenames in sorted(os.walk(root)):
            for filename in sorted(filenames):
                if os.path.splitext(filename)[-1].lower() in self.IMG_EXTENSIONS:
                    path = os.path.join(sub_root, filename)
                    samples.append(path)

        if len(samples) == 0:
            raise RuntimeError(f"Found 0 files in sub-folders of: {root}\n"
                               f"Supported extensions are: {','.join(self.IMG_EXTENSIONS)}")

        self.root = root
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        path = self.samples[index]
        sample = cv2.imread(path)[..., ::-1]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = f'Dataset {self.__class__.__name__}\n'\
                  f'    Number of data points: {self.__len__()}\n'\
                  f'    Root Location: {self.root}\n'
        tmp = '    Transforms (if any): '
        trans_tmp = self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp))
        fmt_str += f'{tmp}{trans_tmp}'
        return fmt_str


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
