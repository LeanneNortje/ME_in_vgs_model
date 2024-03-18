import pdb
import random

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

from show_attention import AUDIO_DIR, IMAGE_DIR, load_vocabulary
from show_phonetic_analysis import l2_norm
from show_tsne import (
    cache_json,
    cache_np,
    compute_audio_embeddings,
    compute_image_embeddings,
    MattNet,
)


seed = 1337
random.seed(seed)


KWARGS_HEATMAP = dict(
    annot=True,
    fmt=".0f",
    vmin=0.0,
    vmax=100.0,
    linewidth=0.5,
    # cmap="bwr",
)


def show_heatmap(df, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, ax=ax, **KWARGS_HEATMAP, **kwargs)
    st.pyplot(fig)


def get_subset_paths(paths, max_num):
    paths = list(paths)
    random.shuffle(paths)
    paths = paths[:max_num]
    paths = [str(path) for path in paths]
    return paths


def get_image_paths(keyword, max_num):
    files = IMAGE_DIR.glob(keyword + "_masked_" + "*")
    return get_subset_paths(files, max_num)


def get_audio_paths(keyword, max_num):
    files = AUDIO_DIR.glob(keyword + "*")
    return get_subset_paths(files, max_num)


def load_data(words):
    return {
        word: {
            "paths-audio": get_audio_paths(word, 30),
            "paths-image": get_image_paths(word, 30),
        }
        for word in words
    }


def compute_similarity_direct(emb_audio, emb_image):
    emb_audio = torch.tensor(emb_audio)
    emb_image = torch.tensor(emb_image)
    sim = torch.einsum("adx,idy->aiy", emb_audio, emb_image)
    sim = sim.max(axis=2).values.numpy()
    return sim


def compute_similarity_audio(emb_audio_1, emb_audio_2):
    def preprocess(emb):
        emb = emb.squeeze(2)
        return l2_norm(emb)

    return preprocess(emb_audio_1) @ preprocess(emb_audio_2).T


def compute_similarity_image(emb_image_1, emb_image_2):
    def preprocess(emb):
        emb = emb.mean(2)
        emb = l2_norm(emb)
        return emb
        # return torch.tensor(emb)

    return preprocess(emb_image_1) @ preprocess(emb_image_2).T
    # emb_image_1 = preprocess(emb_image_1)
    # emb_image_2 = preprocess(emb_image_2)
    # sim = torch.einsum("idu,jdv->ijuv", emb_image_1, emb_image_2)
    # breakpoint()
    # sim = sim.max(axis=2).values.numpy()
    # sim = sim.mean().numpy()
    # return sim


def compute_similarity_direct_words(
    word1, word2, data_audio_flatten, data_image_flatten, emb_audio, emb_image, **kwargs
):
    idxs1 = [i for i, datum in enumerate(data_audio_flatten) if datum["word"] == word1]
    idxs2 = [i for i, datum in enumerate(data_image_flatten) if datum["word"] == word2]
    sim = compute_similarity_direct(emb_audio[idxs1], emb_image[idxs2])
    return sim.mean()


def compute_similarity_audio_words(
    word1, word2, data_audio_flatten, emb_audio, **kwargs
):
    idxs1 = [i for i, datum in enumerate(data_audio_flatten) if datum["word"] == word1]
    idxs2 = [i for i, datum in enumerate(data_audio_flatten) if datum["word"] == word2]
    sim = compute_similarity_audio(emb_audio[idxs1], emb_audio[idxs2])
    return 100 * sim.mean()


def compute_similarity_image_words(
    word1, word2, data_image_flatten, emb_image, **kwargs
):
    idxs1 = [i for i, datum in enumerate(data_image_flatten) if datum["word"] == word1]
    idxs2 = [i for i, datum in enumerate(data_image_flatten) if datum["word"] == word2]
    sim = compute_similarity_image(emb_image[idxs1], emb_image[idxs2])
    return 100 * sim.mean()


def compute_similarity_indirect_words(
    src,
    tgt,
    sim_unseen_audio_to_seen_audio,
    sim_seen_audio_to_seen_image,
    sim_seen_image_to_unseen_image,
    sim_unseen_audio_to_unseen_image,
    words_s,
    **kwargs,
):
    sim = [
        {
            "src": src,
            "tgt": tgt,
            "sim-indirect": (
                sim_unseen_audio_to_seen_audio.loc[src, w1]
                + sim_seen_audio_to_seen_image.loc[w1, w2]
                + sim_seen_image_to_unseen_image.loc[w2, tgt]
            ),
            "sim-direct": sim_unseen_audio_to_unseen_image.loc[src, tgt],
            "sim-unseen-audio-to-seen-audio": sim_unseen_audio_to_seen_audio.loc[
                src, w1
            ],
            "sim-seen-audio-to-seen-image": sim_seen_audio_to_seen_image.loc[w1, w2],
            "sim-seen-image-to-unseen-image": sim_seen_image_to_unseen_image.loc[
                w2, tgt
            ],
            "audio-seen": w1,
            "image-seen": w2,
        }
        for w1 in words_s
        for w2 in words_s
    ]
    result = max(sim, key=lambda x: x["sim-indirect"])
    print(result)
    return result


def compute_unseen_audio_to_unseen_image(*, words_u, **kwargs):
    similarities = [
        {
            "unseen-audio": w1,
            "unseen-image": w2,
            "similarity": compute_similarity_direct_words(w1, w2, **kwargs),
        }
        for w1 in words_u
        for w2 in words_u
    ]

    df = pd.DataFrame(similarities)
    df = df.pivot(index="unseen-audio", columns="unseen-image", values="similarity")

    return df


def compute_seen_audio_to_seen_image(*, words_s, **kwargs):
    similarities = [
        {
            "seen-audio": w1,
            "seen-image": w2,
            "similarity": compute_similarity_direct_words(w1, w2, **kwargs),
        }
        for w1 in words_s
        for w2 in words_s
    ]

    df = pd.DataFrame(similarities)
    df = df.pivot(index="seen-audio", columns="seen-image", values="similarity")

    return df


def compute_unseen_audio_to_seen_audio(*, words_u, words_s, **kwargs):
    similarities = [
        {
            "unseen-audio": w1,
            "seen-audio": w2,
            "similarity": compute_similarity_audio_words(w1, w2, **kwargs),
        }
        for w1 in words_u
        for w2 in words_s
    ]

    df = pd.DataFrame(similarities)
    df = df.pivot(index="unseen-audio", columns="seen-audio", values="similarity")

    return df


def compute_seen_image_to_unseen_image(*, words_s, words_u, **kwargs):
    similarities = [
        {
            "seen-image": w1,
            "unseen-image": w2,
            "similarity": compute_similarity_image_words(w1, w2, **kwargs),
        }
        for w1 in words_s
        for w2 in words_u
    ]

    df = pd.DataFrame(similarities)
    df = df.pivot(index="seen-image", columns="unseen-image", values="similarity")

    return df


def compute_unseen_audio_to_unseen_image_indirect(words_u, **kwargs):
    return [
        compute_similarity_indirect_words(src, tgt, **kwargs)
        for src in words_u
        for tgt in words_u
    ]


def compute_audio_all(words_u, words_s, **kwargs):
    similarities = [
        {
            "audio-1": w1,
            "audio-2": w2,
            "similarity": compute_similarity_audio_words(w1, w2, **kwargs),
        }
        for w1 in words_u + words_s
        for w2 in words_u + words_s
    ]

    df = pd.DataFrame(similarities)
    df = df.pivot(index="audio-1", columns="audio-2", values="similarity")

    words = words_s + words_u
    df = df.reindex(index=words, columns=words)

    return df


def compute_image_all(words_u, words_s, **kwargs):
    similarities = [
        {
            "image-1": w1,
            "image-2": w2,
            "similarity": compute_similarity_image_words(w1, w2, **kwargs),
        }
        for w1 in words_u + words_s
        for w2 in words_u + words_s
    ]

    df = pd.DataFrame(similarities)
    df = df.pivot(index="image-1", columns="image-2", values="similarity")

    words = words_s + words_u
    df = df.reindex(index=words, columns=words)

    return df


def compute_direct_all(words_u, words_s, **kwargs):
    similarities = [
        {
            "image": w1,
            "audio": w2,
            "similarity": compute_similarity_direct_words(w1, w2, **kwargs),
        }
        for w1 in words_u + words_s
        for w2 in words_u + words_s
    ]

    df = pd.DataFrame(similarities)
    df = df.pivot(index="audio", columns="image", values="similarity")

    words = words_s + words_u
    df = df.reindex(index=words, columns=words)

    return df


def load_kwargs():
    model_path = "model_metadata"
    config_name = "base"

    words_s = sorted(load_vocabulary("seen"))
    words_u = sorted(load_vocabulary("unseen"))

    words = words_s + words_u
    data = cache_json("data/indirect-paths-analysis-data.json", load_data, words)

    mattnet = MattNet(config_name, model_path)
    mattnet.eval()

    audio_paths = [path for word in words for path in data[word]["paths-audio"]]
    emb_audio = cache_np(
        f"data/audio-embeddings-indirect-analysis-{config_name}-{seed}.npy",
        compute_audio_embeddings,
        mattnet=mattnet,
        audio_paths=audio_paths,
    )

    image_paths = [path for word in words for path in data[word]["paths-image"]]
    emb_image = cache_np(
        f"data/image-embeddings-indirect-analysis-{config_name}-{seed}.npy",
        compute_image_embeddings,
        mattnet=mattnet,
        image_paths=image_paths,
    )

    data_audio_flatten = [
        {"word": word, "path": path}
        for word in words
        for path in data[word]["paths-audio"]
    ]
    data_image_flatten = [
        {"word": word, "path": path}
        for word in words
        for path in data[word]["paths-image"]
    ]

    return {
        "words_u": words_u,
        "words_s": words_s,
        "data_audio_flatten": data_audio_flatten,
        "data_image_flatten": data_image_flatten,
        "emb_audio": emb_audio,
        "emb_image": emb_image,
    }



def main():
    kwargs = load_kwargs()

    st.markdown("## audio")
    sim_audio = compute_audio_all(**kwargs)
    show_heatmap(sim_audio, cbar=False)

    st.markdown("## image")
    sim_image = compute_image_all(**kwargs)
    show_heatmap(sim_image, cbar=False)

    st.markdown("## direct")
    sim_direct = compute_direct_all(**kwargs)
    show_heatmap(sim_direct, cbar=False)

    st.markdown("## unseen audio vs unseen image")
    sim_unseen_audio_to_unseen_image = compute_unseen_audio_to_unseen_image(**kwargs)
    show_heatmap(sim_unseen_audio_to_unseen_image)

    st.markdown("## unseen audio vs seen audio")
    sim_unseen_audio_to_seen_audio = compute_unseen_audio_to_seen_audio(**kwargs)
    show_heatmap(sim_unseen_audio_to_seen_audio)

    st.markdown("## seen audio vs seen image")
    sim_seen_audio_to_seen_image = compute_seen_audio_to_seen_image(**kwargs)
    show_heatmap(sim_seen_audio_to_seen_image)

    st.markdown("## seen image vs unseen image")
    sim_seen_image_to_unseen_image = compute_seen_image_to_unseen_image(**kwargs)
    show_heatmap(sim_seen_image_to_unseen_image)

    fig, axs = plt.subplots(2, 2, figsize=(20, 20)) # , sharex=True, sharey=True)
    sns.heatmap(sim_unseen_audio_to_seen_audio, ax=axs[0, 0], **KWARGS_HEATMAP, cbar=False)
    sns.heatmap(sim_unseen_audio_to_unseen_image, ax=axs[0, 1], **KWARGS_HEATMAP, cbar=False)
    sns.heatmap(sim_seen_audio_to_seen_image, ax=axs[1, 0], **KWARGS_HEATMAP, cbar=False)
    sns.heatmap(sim_seen_image_to_unseen_image, ax=axs[1, 1], **KWARGS_HEATMAP, cbar=False)

    axs[0, 0].set_title("sim(a', a)")
    axs[0, 1].set_title("sim(a', i') · direct")
    axs[1, 0].set_title("sim(a, i)")
    axs[1, 1].set_title("sim(i, i')")

    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("## indirect path")
    # results_indirect = # cache_json(
    # "output/results-indirect-paths.json",
    results_indirect = compute_unseen_audio_to_unseen_image_indirect(
        words_u=kwargs["words_u"],
        words_s=kwargs["words_s"],
        sim_unseen_audio_to_seen_audio=sim_unseen_audio_to_seen_audio,
        sim_seen_audio_to_seen_image=sim_seen_audio_to_seen_image,
        sim_seen_image_to_unseen_image=sim_seen_image_to_unseen_image,
        sim_unseen_audio_to_unseen_image=sim_unseen_audio_to_unseen_image,
    )
    df = pd.DataFrame(results_indirect)
    df = df.set_index(["src", "tgt"])
    st.dataframe(df)
    df.to_csv("output/results-indirect-paths.csv")


if __name__ == "__main__":
    main()
