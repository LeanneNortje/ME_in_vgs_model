import json
import os
import pdb
import pickle
import random

from collections import OrderedDict
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from sklearn import manifold
from tqdm import tqdm

from torch import nn
from torchdata.datapipes.map import SequenceWrapper
from torchvision import transforms

from models.multimodalModels import mutlimodal as AudioModel
from models.multimodalModels import vision as ImageModel
from models.GeneralModels import ScoringAttentionModule

# from test_ME import (
#     LoadImage as load_image,
#     LoadAudio as load_audio,
#     PadFeat as pad_audio,
# )
from util import *


strip = lambda line: line.strip()
AUDIO_DIR = Path("data/english_words")
IMAGE_DIR = Path("data/images")


def load(path, func):
    with open(path, "r") as f:
        return list(map(func, f.readlines()))


def cache_np(path, func, *args, **kwargs):
    if os.path.exists(path):
        return np.load(path)
    else:
        result = func(*args, **kwargs)
        np.save(path, result)
        return result


def cache_json(path, func, *args, **kwargs):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        result = func(*args, **kwargs)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        return result


CONFIGS = {
    "base": {
        "model-name": "f613809f5c/1",
    },
}


class MattNet(nn.Module):
    def __init__(self, config_name, model_path, device="cpu"):
        super().__init__()

        config = CONFIGS[config_name]
        model_name = config["model-name"]

        self.model_dir = Path(f"{model_path}/{model_name}")

        with open(self.model_dir / "args.pkl", "rb") as f:
            self.args = pickle.load(f)

        self.kwargs_pad_audio = {
            "target_length": self.args["audio_config"].get("target_length", 1024),
            "padval": self.args["audio_config"].get("padval", 0),
        }

        self.img_size = 256, 256
        self.kwargs_load_image = {
            "resize": transforms.Resize(self.img_size),
            "to_tensor": transforms.ToTensor(),
            "image_normalize": transforms.Normalize(
                mean=self.args["image_config"]["RGB_mean"],
                std=self.args["image_config"]["RGB_std"],
            ),
        }

        audio_model = AudioModel(self.args)
        image_model = ImageModel(self.args)
        attention_model = ScoringAttentionModule(self.args)

        path_checkpoint = self.model_dir / "models" / "best_ckpt.pt"
        state = torch.load(path_checkpoint, map_location=device)

        # pdb.set_trace()
        audio_model.load_state_dict(self.fix_ddp_module(state["english_model"]))
        image_model.load_state_dict(self.fix_ddp_module(state["image_model"]))
        attention_model.load_state_dict(self.fix_ddp_module(state["attention"]))

        self.audio_model = audio_model
        self.image_model = image_model
        self.attention_model = attention_model

    @staticmethod
    def fix_ddp_module(state):
        # remove 'module.' of DistributedDataParallel (DDP)
        def rm_prefix(key):
            SEP = "."
            prefix, *rest = key.split(SEP)
            assert prefix == "module"
            return SEP.join(rest)

        return OrderedDict([(rm_prefix(k), v) for k, v in state.items()])

    def forward(self, audio, image):
        image_emb = self.image_model(image.unsqueeze(0))
        # image_emb = image_emb.view(image_emb.size(0), image_emb.size(1), -1)
        # image_emb = image_emb.transpose(1, 2)
        _, _, audio_emb = self.audio_model(audio)
        att = self.attention_model.get_attention(image_emb, audio_emb)
        score = att.max()
        return score, att

    def load_image_1(self, image_path):
        image = load_image(image_path, **self.kwargs_load_image)
        return image

    def load_audio_1(self, audio_path):
        audio, _ = load_audio(audio_path, self.args["audio_config"])
        audio, _ = pad_audio(audio, **self.kwargs_pad_audio)
        return audio


def load_vocabulary(type):
    return load(f"data/{type}.txt", strip)


def get_audio_path(filename):
    return AUDIO_DIR / (filename + ".wav")


def get_image_path(filename):
    return IMAGE_DIR / (filename + ".jpg")


def is_masked(filename) -> bool:
    _, masked, *_ = filename.split("_")
    return masked == "masked"


def extract_word(filename) -> str:
    word, *_ = filename.split("_")
    return word


def compute_audio_embeddings(mattnet, audio_paths):
    batch_size = 100

    dp = SequenceWrapper(audio_paths)
    dp = dp.map(mattnet.load_audio_1)
    dp = dp.batch(batch_size=batch_size)

    with torch.no_grad():
        audio_embeddings = (
            mattnet.audio_model(torch.cat(batch, dim=0))[2] for batch in tqdm(dp)
        )
        audio_embeddings = np.concatenate([e.numpy() for e in audio_embeddings])

    return audio_embeddings


def compute_image_embeddings(mattnet, image_paths):
    batch_size = 100

    dp = SequenceWrapper(image_paths)
    dp = dp.map(mattnet.load_image_1)
    dp = dp.batch(batch_size=batch_size)

    with torch.no_grad():
        image_embeddings = (
            mattnet.image_model(torch.stack(batch)) for batch in tqdm(dp)
        )
        image_embeddings = np.concatenate([e.numpy() for e in image_embeddings])

    return image_embeddings


def show(df):
    fig, ax = plt.subplots()
    sns.scatterplot(df, x="x", y="y", hue="is-seen", style="word", ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def load_test_samples():
    episodes = np.load(Path("data/episodes.npz"), allow_pickle=True)["episodes"].item()
    image_names = set()
    audio_names = set()

    for num in episodes:
        test = "novel_test"
        for sample in episodes[num][test]:
            for c, im, aud, dataset in episodes[num][test][sample]:
                if is_masked(str(im)):
                    image_names.add(im.stem)
                audio_names.add(aud.stem)

    development = np.load(Path("data/val_lookup.npz"), allow_pickle=True)[
        "lookup"
    ].item()
    for c in development:
        for im in development[c]["images"]:
            if is_masked(str(im)):
                image_names.add(im)

        for im in development[c]["english"]:
            audio_names.add(aud)

    return image_names, audio_names


@click.command()
@click.option(
    "-c", "--config", "config_name", required=True, type=click.Choice(CONFIGS)
)
@click.option("-p", "--path", "model_path", required=False, default="output/models")
def main(config_name, model_path):
    image_names, audio_names = load_test_samples()
    # image_names = load("data/filelist-image.txt", strip)
    # image_names = [i for i in image_names if is_masked(i)]
    image_paths = [get_image_path(i) for i in image_names]

    words = [extract_word(i) for i in image_names]
    words_seen = load_vocabulary("seen")
    words_unseen = load_vocabulary("unseen")
    are_seen = [w in words_seen for w in words]

    mattnet = MattNet(config_name, model_path)
    mattnet.eval()

    emb = cache_np(
        f"data/image_embeddings_masked_{config_name}.npy",
        compute_image_embeddings,
        mattnet=mattnet,
        image_paths=image_paths,
    )

    # TSNE
    t_sne = manifold.TSNE(
        # perplexity=30,
        # init="random",
        # n_iter=1000,
        # random_state=0,
    )

    emb = emb.mean(axis=-1)
    # emb = emb.sum(axis=-1)
    # emb = np.reshape(emb, (emb.shape[0], -1))
    emb2d = t_sne.fit_transform(emb)
    x, y = emb2d.T

    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "word": words,
            "is-seen": are_seen,
        }
    )

    st.set_page_config(layout="wide")
    # st.write(df)

    words_seen = words_seen[:8]
    words_unseen = words_unseen[:8]

    num_seen = len(words_seen)
    num_unseen = len(words_unseen)

    def make_scatter_images(ax, idxs, to_show_legend=False):
        sns.scatterplot(
            df,
            x="x",
            y="y",
            hue="is-seen",
            style="word",
            legend=to_show_legend,
            ax=ax,
        )
        if to_show_legend:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        for i in idxs:
            arr_hand = plt.imread(image_paths[i])
            imagebox = OffsetImage(arr_hand, zoom=0.05)
            xy = x[i], y[i]
            color = "orange" if are_seen[i] else "blue"

            ab = AnnotationBbox(
                imagebox,
                xy,
                pad=0.1,
                bboxprops={"color": color, "linewidth": 0.4},
            )
            ax.add_artist(ab)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    idxs_all = list(range(len(df)))
    nrows = 6
    sz = 7
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(sz, sz * nrows))
    axs = iter(axs)

    ax = next(axs)
    make_scatter_images(ax, [], to_show_legend=True)
    ax.set_title("scatter plot")

    ax = next(axs)
    idxs_seen = [i for i, s in enumerate(are_seen) if s]
    idxs = random.sample(idxs_seen, 75)
    random.shuffle(idxs)
    make_scatter_images(ax, idxs)
    ax.set_title("images (seen only)")

    ax = next(axs)
    idxs_unseen = [i for i, s in enumerate(are_seen) if not s]
    idxs = random.sample(idxs_unseen, 75)
    random.shuffle(idxs)
    make_scatter_images(ax, idxs)
    ax.set_title("images (unseen only)")

    for i in range(3):
        ax = next(axs)
        idxs = random.sample(idxs_all, 75)
        random.shuffle(idxs)
        make_scatter_images(ax, idxs)
        ax.set_title(f"images (run {i + 1})")

    fig.tight_layout()
    st.pyplot(fig)

    # fig, axs = plt.subplots(
    #     num_seen,
    #     num_unseen,
    #     figsize=(num_seen * 2.5, num_unseen * 2.5),
    #     sharex=True,
    #     sharey=True,
    # )

    # for r, w1 in enumerate(words_seen):
    #     for c, w2 in enumerate(words_unseen):
    #         idxs1 = df["word"] == w1
    #         idxs2 = df["word"] == w2
    #         idxs = idxs1 | idxs2

    #         ax = axs[r, c]
    #         sns.scatterplot(
    #             df[idxs],
    #             x="x",
    #             y="y",
    #             hue="is-seen",
    #             ax=ax,
    #         )

    #         ax.set_title(f"seen: {w1} | unseen: {w2}")
    #         ax.set_xticklabels([])
    #         ax.set_yticklabels([])
    #         ax.set_xticks([])
    #         ax.set_yticks([])

    # fig.tight_layout()
    # st.pyplot(fig)

    # audios = load("data/filelist-audio.txt", strip)
    # for _ in range(32):
    #     audio = random.choice(audios)
    #     image = random.choice(images)
    #     st.write(audio, image)
    #     st.audio(str(get_audio_path(audio)))
    #     st.image(str(get_image_path(image)))
    #     st.markdown("---")


if __name__ == "__main__":
    main()
