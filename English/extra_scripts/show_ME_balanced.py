#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2023
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________
# adapted from https://github.com/dharwath

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel as DDP
from dataloaders import *
from models.setup import *
from models.util import *
from models.GeneralModels import *
from models.multimodalModels import *
from training.util import *
from evaluation.calculations import *
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from training import validate
import time
from tqdm import tqdm

import numpy as trainable_parameters
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy
import scipy.signal
from scipy.spatial import distance
import librosa
import matplotlib.lines as lines

import itertools
import seaborn as sns
from torchvision.io import read_image
from torchvision.models import *

from PIL import Image
from matplotlib import image
from matplotlib import pyplot


results = np.load(Path('results/balanced_per_keyword_results.npz'), allow_pickle=True)['per_keyword_results'].item()
totals = np.load(Path('results/balanced_per_keyword_results.npz'), allow_pickle=True)['totals'].item()

    
overall_c = 0
overall_t = 0
scores = []
for novel_label in results:
    print()
    print(novel_label)
    t = totals[novel_label]
    check = 0

    for label in results[novel_label]:
        n = results[novel_label][label]
        p = round(n/t * 100, 2)
        print(f'{label:<10}\t{n}\t{t} = {p}%')
        check += n
        if label == novel_label: 
            overall_c += n
            scores.append(p)
    overall_t += t
    print(f'Check: {check}')

p = round(overall_c/overall_t * 100, 2)
print(f'Overall accuracy: {overall_c} / {overall_t} = {p}%')

m = np.mean(scores)
v = np.std(scores)
print(m, v)
# np.savez(
#     Path('results/balanced_per_keyword_results'),
#     per_keyword_results=results,
#     totals=totals
#     )