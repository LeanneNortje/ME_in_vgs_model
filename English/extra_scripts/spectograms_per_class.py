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


config_library = {
    "multilingual": "English_Hindi_DAVEnet_config.json",
    "multilingual+matchmap": "English_Hindi_matchmap_DAVEnet_config.json",
    "english": "English_DAVEnet_config.json",
    "english+matchmap": "English_matchmap_DAVEnet_config.json",
    "hindi": "Hindi_DAVEnet_config.json",
    "hindi+matchmap": "Hindi_matchmap_DAVEnet_config.json",
}

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

def preemphasis(signal,coeff=0.97):  
    # function adapted from https://github.com/dharwath
    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def LoadAudio(path, audio_conf):
    
    audio_type = audio_conf.get('audio_type')
    if audio_type not in ['melspectrogram', 'spectrogram']:
        raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')

    preemph_coef = audio_conf.get('preemph_coef')
    sample_rate = audio_conf.get('sample_rate')
    window_size = audio_conf.get('window_size')
    window_stride = audio_conf.get('window_stride')
    window_type = audio_conf.get('window_type')
    num_mel_bins = audio_conf.get('num_mel_bins')
    target_length = audio_conf.get('target_length')
    fmin = audio_conf.get('fmin')
    n_fft = audio_conf.get('n_fft', int(sample_rate * window_size))
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    # load audio, subtract DC, preemphasis
    y, sr = librosa.load(path, sample_rate)
    dur = librosa.get_duration(y=y, sr=sr)
    nsamples = y.shape[0]
    if y.size == 0:
        y = np.zeros(target_length)
    y = y - y.mean()
    y = preemphasis(y, preemph_coef)

    # compute mel spectrogram / filterbanks
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=scipy_windows.get(window_type, scipy_windows['hamming']))
    spec = np.abs(stft)**2 # Power spectrum
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
        melspec = np.dot(mel_basis, spec)
        logspec = librosa.power_to_db(melspec, ref=np.max)
    elif audio_type == 'spectrogram':
        logspec = librosa.power_to_db(spec, ref=np.max)
    # n_frames = logspec.shape[1]
    logspec = torch.FloatTensor(logspec)
    nsamples = logspec.size(1)
    # print(logspec.size())
    return logspec#, nsamples#, n_frames

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--config-file", type=str, default='matchmap', choices=['matchmap'], help="Model config file.")
parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
parser.add_argument("--image-base", default="/storage", help="Model config file.")
command_line_args = parser.parse_args()
restore_epoch = command_line_args.restore_epoch

args, image_base = modelSetup(command_line_args, True)

audio_conf = args["audio_config"]


episodes = np.load(Path('data/episodes.npz'), allow_pickle=True)['episodes'].item()

audio_datapoints = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_datapoints'].item()
# image_datapoints = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_datapoints'].item()

query = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_1'].item()
query_labels = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_labels_1'].item()
query_tag = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_tag_1'].item()

secondary_query = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_2'].item()
secondary_labels = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_labels_2'].item()
secondary_tag = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['audio_tag_2'].item()

# image_1 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_1'].item()
# image_2 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_2'].item()
# image_labels_1 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_labels_1'].item()
# image_labels_2 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_labels_2'].item()
# image_tag_1 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_tag_1'].item()
# image_tag_2 = np.load(Path('results/files/episode_data.npz'), allow_pickle=True)['image_tag_2'].item()

class_wavs = {}
novels = set()
familiars = set()
for ep_num in episodes:
    for i, wav in enumerate(query[ep_num]):
        c = Path(wav).stem.split('_')[0]
        if c not in class_wavs: class_wavs[c] = set()
        class_wavs[c].add(wav)

        if query_tag[ep_num][i] == 'novel': novels.add(query_labels[ep_num][i])
        else: familiars.add(query_labels[ep_num][i])

# preemph_coef = audio_conf.get('preemph_coef')
sample_rate = audio_conf.get('sample_rate')
window_size = audio_conf.get('window_size')
# window_stride = audio_conf.get('window_stride')
# window_type = audio_conf.get('window_type')
# num_mel_bins = audio_conf.get('num_mel_bins')
# target_length = audio_conf.get('target_length')
# fmin = audio_conf.get('fmin')
n_fft = audio_conf.get('n_fft', int(sample_rate * window_size))
win_length = int(sample_rate * window_size)
# hop_length = int(sample_rate * window_stride)
num_per_class = 3
audio_conf['audio_type'] = 'spectrogram'
for c in class_wavs:
    print(c, c in novels, c in familiars, len(class_wavs[c]))

    fig, plots = plt.subplots(nrows=num_per_class, gridspec_kw={'width_ratios': [100], 'height_ratios': [0.5, 0.5, 0.5]})
    # fig.set_figheight(5)
    # fig.set_figwidth(5)
    for i in range(num_per_class):
        # y, sr = librosa.load(list(class_wavs[c])[i], audio_conf['sample_rate'])
        # spec, freqs, bins, im = plots[i].specgram(np.asarray(y), NFFT=n_fft, Fs=sample_rate, window=np.hamming(win_length))
        spec = LoadAudio(list(class_wavs[c])[i], audio_conf)
        frames = np.flip(spec.numpy()).shape[-1]
        plots[i].imshow(np.flip(spec.numpy(), axis=0), extent=[0, frames//10, 0, 1])
        plots[i].axis('off')
    
    if c in novels: 
        fig.suptitle(f'Novel class: {c}', fontsize=20)
        plt.savefig(f'examples/novel_{c}.pdf',bbox_inches='tight')
        plt.savefig(f'examples/novel_{c}.png',bbox_inches='tight')
    elif c in familiars: 
        fig.suptitle(f'Familiar class: {c}', fontsize=20)
        plt.savefig(f'examples/familiar_{c}.pdf',bbox_inches='tight')
        plt.savefig(f'examples/familiar_{c}.png',bbox_inches='tight')
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
