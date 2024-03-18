import pdb

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# sns.set_context("talk")


def load_data():
    df = pd.read_excel(
        "results/sheet.xlsx",
        sheet_name="Keyword analysis",
        header=1,
    )
    df.columns = ["query", "foil", "total", "selected", "prop-1", "prop", "prop-comp"]
    # df = df.drop(columns=["selected", "prop-1", "prop", "prop-comp"])
    df = df.drop(columns=["total", "selected", "prop-1", "prop-comp"])
    df = df.dropna(axis=0, how="any", subset=["query"])
    return df


def load_mismatched_data():
    df = pd.read_excel(
        "results/sheet.xlsx",
        sheet_name="Mismatched keyword analysis",
        header=1,
    )
    df.columns = ["query", "foil", "total", "selected", "prop-1", "prop", "prop-comp"]
    # df = df.drop(columns=["selected", "prop-1", "prop", "prop-comp"])
    df = df.drop(columns=["total", "selected", "prop-1", "prop-comp"])
    df = df.dropna(axis=0, how="any", subset=["query"])
    return df


def show_me_per_class(df, ax):
    df0 = df[df["query"] == df["foil"]]
    df0 = df0.sort_values(by="query")

    # sns.set_theme(style="whitegrid")
    # sns.barplot(data=df0, x="prop", y="query", ax=ax)
    # sns.despine(left=True, bottom=True, ax=ax)
    with sns.axes_style("whitegrid"):
        sns.stripplot(df0, x="prop", y="query", jitter=False, orient="h", linewidth=1, size=5, color="black", ax=ax)
        sns.despine(ax=ax)
        ax.axvline(50, color="red", clip_on=False, zorder=1, linewidth=1, linestyle="--")
        ax.set(xlabel="Mutual exclusivity · Accuracy (%)", ylabel="Word · Novel audio", xlim=(0, 100))
        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)


def show_me_pairwise(df, ax):
    df1 = df[df["query"] != df["foil"]]
    df1 = df1.replace("#DIV/0!", np.nan)
    df1 = df1.reset_index(drop=True)

    # df1["total"] = df1["total"] / 1000

    # print(df1)
    # print(df1.iloc[28])
    # df1

    df1 = df1.pivot(index="query", columns="foil", values="prop")
    # df = df.pivot(index="query", columns="foil", values="total")
    # fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        df1,
        annot=True,
        fmt=".0f",
        vmin=0,
        vmax=100,
        linewidth=0.5,
        cmap="bwr",
        cbar=False,
        square=True,
        annot_kws={"fontsize": 8},
        ax=ax,
    )
    ax.set(xlabel="Familiar image", ylabel="Novel audio")
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)


def main():
    df = load_data()
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    show_me_per_class(df, axs[0])
    show_me_pairwise(df, axs[1])
    fig.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
