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
import csv

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
from sklearn.preprocessing import normalize

BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"

flickr_boundaries_fn = Path('/storage/Datasets/flickr_audio/flickr_8k.ctm')
flickr_audio_dir = flickr_boundaries_fn.parent / "wavs"
flickr_images_fn = Path('/storage/Datasets/Flicker8k_Dataset/')
flickr_segs_fn = Path('./data/flickr_image_masks/')

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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
parser.add_argument("--config-file", type=str, default='matchmap', choices=['matchmap'], help="Model config file.")
parser.add_argument("--restore-epoch", type=int, default=-1, help="Epoch to generate accuracies for.")
parser.add_argument("--image-base", default="/storage", help="Model config file.")
command_line_args = parser.parse_args()
restore_epoch = command_line_args.restore_epoch

# Setting up model specifics
heading(f'\nSetting up model files ')
args, image_base = modelSetup(command_line_args, True)
rank = 'cuda'

audio_conf = args["audio_config"]
target_length = audio_conf.get('target_length', 128)
padval = audio_conf.get('padval', 0)
image_conf = args["image_config"]
crop_size = image_conf.get('crop_size')
center_crop = image_conf.get('center_crop')
RGB_mean = image_conf.get('RGB_mean')
RGB_std = image_conf.get('RGB_std')

# image_resize_and_crop = transforms.Compose(
#         [transforms.Resize(224), transforms.ToTensor()])
resize = transforms.Resize((256, 256))
to_tensor = transforms.ToTensor()
image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

image_resize = transforms.transforms.Resize((256, 256))
trans = transforms.ToPILImage()

# Create models
english_model = mutlimodal(args).to(rank)

if rank == 0: heading(f'\nSetting up image model ')
image_model = vision(args).to(rank)

if rank == 0: heading(f'\nSetting up attention model ')
attention = ScoringAttentionModule(args).to(rank)

if rank == 0: heading(f'\nSetting up contrastive loss ')
contrastive_loss = ContrastiveLoss(args).to(rank)


model_with_params_to_update = {
    "enlish_model": english_model,
    "attention": attention,
    "contrastive_loss": contrastive_loss,
    "image_model": image_model
    }
model_to_freeze = {
    }
trainable_parameters = getParameters(model_with_params_to_update, model_to_freeze, args)

if args["optimizer"] == 'sgd':
    optimizer = torch.optim.SGD(
        trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
        momentum=args["momentum"], weight_decay=args["weight_decay"]
        )
elif args["optimizer"] == 'adam':
    optimizer = torch.optim.Adam(
        trainable_parameters, args["learning_rate_scheduler"]["initial_learning_rate"],
        weight_decay=args["weight_decay"]
        )
else:
    raise ValueError('Optimizer %s is not supported' % args["optimizer"])


english_model = DDP(english_model, device_ids=[rank])
image_model = DDP(image_model, device_ids=[rank]) 

if "restore_epoch" in args:
    info, start_epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAtEpochAMP(
        args["exp_dir"], english_model, image_model, attention, contrastive_loss, optimizer, rank, 
        args["restore_epoch"]
        )
else: 
    heading(f'\nRetoring model parameters from best epoch ')
    info, epoch, global_step, best_epoch, best_acc = loadModelAttriburesAndTrainingAMP(
        args["exp_dir"], english_model, image_model, attention, contrastive_loss, optimizer, 
        rank, False
        )

audio_datapoints = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_datapoints'].item()
image_datapoints = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_datapoints'].item()

query = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_1'].item()
query_labels = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_labels_1'].item()
query_tag = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_tag_1'].item()

secondary_query = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_2'].item()
secondary_labels = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_labels_2'].item()
secondary_tag = np.load(Path('results/episode_data.npz'), allow_pickle=True)['audio_tag_2'].item()

image_1 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_1'].item()
image_2 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_2'].item()
image_labels_1 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_labels_1'].item()
image_labels_2 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_labels_2'].item()
image_tag_1 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_tag_1'].item()
image_tag_2 = np.load(Path('results/episode_data.npz'), allow_pickle=True)['image_tag_2'].item()

episodes = np.load(Path('data/episodes.npz'), allow_pickle=True)['episodes'].item()
significance_dict = {}
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

with torch.no_grad():
    for ep_num in tqdm(episodes):

        if ep_num not in significance_dict: significance_dict[ep_num] = {}
        episode = episodes[ep_num]

        tags_1 = query_tag[ep_num]
        labels_1 = query_labels[ep_num]
        tags_2 = secondary_tag[ep_num]
        labels_2 = secondary_labels[ep_num]

        query_output = []
        query_audio_samples = []
        for n, name in enumerate(query[ep_num]):
            if tags_1[n] == 'novel':
                query_audio_samples.append(name.stem)
                query_output.append(audio_datapoints[name])
        query_output= torch.cat(query_output, dim=0)
        _, _, query_output = english_model(query_output.to(rank))


        query_image = []
        query_image_samples = []
        for n, name in enumerate(image_1[ep_num]):
            if tags_1[n] == 'novel':
                query_image_samples.append(name.stem)
                query_image.append(image_datapoints[name])
        query_image = torch.cat(query_image, dim=0)
        query_image = image_model(query_image.to(rank))
        
        indices = [i for i in range(len(query_image_samples))]
        indices.append(indices[0])
        indices.pop(0)

        temp = query_image[indices]
        query_image = temp

        query_image_samples = [query_image_samples[ind] for ind in indices]

#             secondary_query_output = []
#             for name in secondary_query[ep_num]:
#                 secondary_query_output.append(audio_datapoints[name])
#             secondary_query_output= torch.cat(secondary_query_output, dim=0)
#             _, _, secondary_query_output = english_model(secondary_query_output.to(rank))

        other_image = []
        other_image_samples = []
        for n, name in enumerate(image_2[ep_num]):
            if tags_2[n] == 'known_novel':
                other_image_samples.append(name.stem)
                other_image.append(image_datapoints[name])
        other_image = torch.cat(other_image, dim=0)
        other_image = image_model(other_image.to(rank))




        for i in range(query_output.size(0)):

            images = torch.cat([query_image[i, :, :].unsqueeze(0), other_image[i, :, :].unsqueeze(0)], dim=0)
    
            scores = attention.one_to_many_score(images, query_output[i, :, :].unsqueeze(0)).squeeze()

            im = attention.attention_scores(query_image[i, :, :].unsqueeze(0), query_output[i, :, :].unsqueeze(0)).squeeze()
            # im = query_image[i, :, :].unsqueeze(0).mean(-1).squeeze()   
            distance_1 = 1 - cos(im.squeeze().unsqueeze(0), query_output[i, :, :].squeeze().unsqueeze(0)).item()

            im = attention.attention_scores(other_image[i, :, :].unsqueeze(0), query_output[i, :, :].unsqueeze(0)).squeeze() 
            # im = other_image[i, :, :].unsqueeze(0).mean(-1).squeeze()              
            distance_2 = 1 - cos(im.squeeze().unsqueeze(0), query_output[i, :, :].squeeze().unsqueeze(0)).item()
            distances = [distance_1, distance_2]
            index = torch.argmax(scores).item()

            if query_audio_samples[i] not in significance_dict[ep_num]: 
                significance_dict[ep_num][query_audio_samples[i]] = {
                    'image_A': query_image_samples[i],
                    'image_B': other_image_samples[i],
                    'similarity_A': scores[0].item(),
                    'similarity_B': scores[1].item(),
                    'distance_A': distance_1,
                    'distance_B': distance_2
                    }

    

header = ['trial', 'audio_query', 'image_A', 'image_B', 'similarity_A', 'similarity_B', 'distance_A', 'distance_B']
trial = 1
name = 'me_mismatched_analysis'
if args['cpc']['warm_start']: name += '_cpc'
if args['pretrained_alexnet']: name += '_alexnet'
name += f'_{args["instance"]}'
with open(f'results/files/{name}.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for num in tqdm(significance_dict, desc=f'Writing to csv'):
        for query in significance_dict[num]:
            entry = significance_dict[num][query]
            row = [trial, query, entry['image_A'], entry['image_B'], entry['similarity_A'], entry['similarity_B'], entry['distance_A'], entry['distance_B']]
            writer.writerow(row)
            trial += 1
