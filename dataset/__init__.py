import torch
import torchvision.transforms as trn
import h5py
import numpy as np

from torch.utils.data import Dataset
from demo.predictor import COCODemo
from maskrcnn_benchmark.config import cfg


class CaptionDataset(Dataset):
    def __init__(self, input_file, use_maskrcnn_benchmark=True, transform=None):
        h = h5py.File(input_file)
        self.imgs = h['images']
        self.captions = h['captions']
        self.captions_per_img = h.attrs['captions_per_image']
        self.coco_demo = COCODemo(cfg)
        assert self.captions.shape[0] // self.imgs.shape[0] == self.captions_per_img

        if transform is not None:
            # if customer transform rules are defined
            # we will use this
            self.transform = transform
        elif use_maskrcnn_benchmark:
            # if we use maskrcnn_benchmark as our encoder
            # we need to follow the corresponding image
            # pre-process procedure
            self.transform = self.coco_demo.build_transform()
        else:
            self.transform = trn.Compose([trn.Resize(255), trn.ToTensor()])

        assert self.imgs.shape[0] * 1 == self.captions.shape[0]

    def __getitem__(self, item):
        img = self.imgs[item // self.captions_per_img]
        img = self.transform(img)

        caption = self.captions[item]
        caption = torch.from_numpy(caption).long()

        data = {'image': img, 'caption': caption}
        return data

    def __len__(self):
        return self.captions.shape[0]
