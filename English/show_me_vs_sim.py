import pdb
import random

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

import show_proportion_selected as me
import show_indirect as sim


sns.set(font="Arial", style="whitegrid")


KWARGS_HEATMAP = dict(
    annot=True,
    fmt=".0f",
    vmin=0.0,
    vmax=100.0,
    linewidth=0.5,
    cbar=False,
    square=True,
    # cmap="bwr",
)

# # fig, axs = plt.subplots(1, 4, figsize=(17, 6))
# fig, axs = plt.subplots(1, 2, figsize=(9, 6))
# df0 = me.load_data()

# me.show_me_per_class(df0, axs[0])
# me.show_me_pairwise(df0, axs[1])


# # kwargs = sim.load_kwargs()
# # df1 = sim.compute_unseen_audio_to_seen_audio(**kwargs)
# # df2 = sim.compute_seen_image_to_unseen_image(**kwargs)

# # sns.heatmap(df1, **KWARGS_HEATMAP, ax=axs[2], annot_kws={"fontsize":8})
# # # sns.heatmap(df2.T, **KWARGS_HEATMAP, ax=axs[3])

# axs[0].set(xlabel="Accuracy (%)", ylabel="Word", title="Mutual exclusivity\nNovel class preference (%)")
# axs[1].set(xlabel="Familiar class", ylabel="Novel class", title="Familiar class preference (%)")
# # axs[2].set(xlabel="Familiar class", ylabel="Novel class", title="Audio similarities")
# # axs[3].set(xlabel="Familiar class", ylabel="Novel class", title="Image similarities")

# fig.tight_layout()
# fig.savefig("results/figures/me-vs-sim.pdf")
# st.pyplot(fig)


# fig, axs = plt.subplots(1, 2, figsize=(9, 6))
# df1 = me.load_mismatched_data()
# me.show_me_per_class(df1, axs[0])
# me.show_me_pairwise(df1, axs[1])

# axs[0].set(xlabel="Accuracy (%)", ylabel="Word", title="Mutual exclusivity\nNovel class preference (%)")
# axs[1].set(xlabel="Familiar class", ylabel="Novel class", title="Familiar class preference (%)")
# # axs[2].set(xlabel="Familiar class", ylabel="Novel class", title="Audio similarities")
# # axs[3].set(xlabel="Familiar class", ylabel="Novel class", title="Image similarities")

# fig.tight_layout()
# fig.savefig("results/figures/mismatched-me-vs-sim.pdf")
# st.pyplot(fig)



df0 = me.load_data()

fig, axs = plt.subplots(1, 1, figsize=(5, 6))
me.show_me_per_class(df0, axs)
axs.set(xlabel="Accuracy (%)", ylabel="Word", title="Mutual exclusivity\nNovel class preference (%)")
fig.tight_layout()
fig.savefig("results/figures/per_word_me.pdf")
st.pyplot(fig)

fig, axs = plt.subplots(1, 1, figsize=(5, 6))
me.show_me_pairwise(df0, axs)
axs.set(xlabel="Familiar class", ylabel="Novel class", title="Familiar class preference (%)")
fig.tight_layout()
fig.savefig("results/figures/per_word_me_confusion_matrix.pdf")
st.pyplot(fig)

fig, axs = plt.subplots(1, 1, figsize=(5, 6))
kwargs = sim.load_kwargs()
df1 = sim.compute_unseen_audio_to_seen_audio(**kwargs)
sns.heatmap(df1, **KWARGS_HEATMAP, ax=axs, annot_kws={"fontsize":8})
axs.set(xlabel="Familiar class", ylabel="Novel class", title="Audio similarities")
# axs[3].set(xlabel="Familiar class", ylabel="Novel class", title="Image similarities")

fig.tight_layout()
fig.savefig("results/figures/per_word_me_audio_confusion_matrix.pdf")
st.pyplot(fig)


