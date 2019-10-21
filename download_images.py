import torch
import os
import argparse
import json
import wget
from utils import util

from models import ShowTellModel
from Data import CaptionDataset
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread
import torchvision.transforms as trn

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu



dir_main = os.path.abspath(os.path.join(__file__, "../.."))
embed_dim = 512  # default embedding size

# PATH = 'encoder.pth'


if __name__ == '__main__':
    PATH = 'n07729384'
    os.mkdir(PATH)
    with open('n07729384.txt') as f:
        for idx, line in enumerate(f.readlines()):
            try:
                wget.download(line, os.path.join(PATH, 'img_{}.jpg'.format(idx)))
            except:
                continue

