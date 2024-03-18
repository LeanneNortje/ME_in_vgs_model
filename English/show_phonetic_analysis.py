import pdb
import random

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from show_attention import AUDIO_DIR, load_vocabulary
from show_tsne import cache_json, cache_np, compute_audio_embeddings, MattNet


seed = 1337
random.seed(seed)


def get_audio_paths(keyword, max_num):
    files = AUDIO_DIR.glob(keyword + "*")
    files = list(files)
    random.shuffle(files)
    return files[:max_num]


def load_data_1(type):
    return [
        {"word": word, "type": type, "path": str(path)}
        for word in sorted(load_vocabulary(type))
        for path in get_audio_paths(word, 30)
    ]


def load_data():
    return load_data_1("seen") + load_data_1("unseen")


def l2_norm(X):
    """L2 normalization of N x D matrix."""
    mag = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    return X / mag


def main():
    model_path = "output/models"
    config_name = "base"

    words_s = sorted(load_vocabulary("seen"))
    words_u = sorted(load_vocabulary("unseen"))

    data = cache_json("data/phonetic-analysis-data.json", load_data)
    audio_paths = [datum["path"] for datum in data]

    mattnet = MattNet(config_name, model_path)
    mattnet.eval()

    def compute_similarity(w1, w2, emb):
        idxs1 = [i for i, datum in enumerate(data) if datum["word"] == w1]
        idxs2 = [i for i, datum in enumerate(data) if datum["word"] == w2]
        sims = emb[idxs1] @ emb[idxs2].T
        return np.mean(sims)

    emb = cache_np(
        f"data/audio-embeddings-phonetic-analysis-{config_name}-{seed}.npy",
        compute_audio_embeddings,
        mattnet=mattnet,
        audio_paths=audio_paths,
    )
    emb = emb.squeeze(2)
    emb = l2_norm(emb)
    similarities = [
        {
            "word-seen": w1,
            "word-unseen": w2,
            "similarity": compute_similarity(w1, w2, emb),
        }
        for w1 in words_s
        for w2 in words_u
    ]
    df = pd.DataFrame(similarities)
    print(df)

    df = df.pivot(index="word-unseen", columns="word-seen", values="similarity")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        linewidth=0.5,
        # cmap="bwr",
        ax=ax,
    )
    st.pyplot(fig)



if __name__ == "__main__":
    main()
