import random
import pdb

import numpy as np
import streamlit as st
import torch

from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from show_tsne import MattNet, AUDIO_DIR, IMAGE_DIR, load_vocabulary


def get_audio_path(keyword):
    files = AUDIO_DIR.glob(keyword + "*")
    return random.choice(list(files))


def get_image_path(keyword):
    files = IMAGE_DIR.glob(keyword + "_masked_" + "*")
    return random.choice(list(files))


def do(mattnet, audio_path, image_path):
    audio = mattnet.load_audio_1(audio_path)
    image = mattnet.load_image_1(image_path)
    with torch.no_grad():
        score, att = mattnet(audio, image)
    return {
        "score": score,
        "att": att,
    }


def main():
    model_path = "model_metadata" #"output/models" #
    config_name = "base"

    words_s = sorted(load_vocabulary("seen"))
    words_u = sorted(load_vocabulary("unseen"))

    with st.sidebar:
        keyword1 = st.selectbox("Keyword", words_u, words_u.index("nautilus"))
        keyword2 = st.selectbox("Keyword", words_s, words_s.index("elephant"))
        num_episodes = st.number_input("Number of episodes", min_value=1, value=5, step=1)
    
    mattnet = MattNet(config_name, model_path)
    mattnet.eval()

    keywords = [keyword1, keyword2]

    def get_image_explanation(result):
        attention = result["att"]
        attention = attention.view(7, 7)
        attention = 5 * (attention / 100 - 0.5)
        explanation = torch.sigmoid(attention).numpy()

        image_rgb = Image.open(result["image-path"])
        image_rgb = np.array(image_rgb) / 255
        h, w, _ = image_rgb.shape

        explanation = Image.fromarray(explanation).resize((w, h))
        explanation = np.array(explanation)
        return show_cam_on_image(image_rgb, explanation, use_rgb=True)

    for _ in range(num_episodes):
        audio_paths = list(map(get_audio_path, keywords))
        image_paths = list(map(get_image_path, keywords))

        results = [
            [
                {
                    **do(mattnet, audio_path, image_path),
                    "audio-path": audio_path,
                    "image-path": image_path,
                }
                for image_path in image_paths
            ]
            for audio_path in audio_paths
        ]

        for r in range(2):
            columns = st.columns(3)
            with columns[0]:
                st.markdown("query: " + keywords[r])
                st.audio(str(results[r][0]["audio-path"]))

            scores = [results[r][c]["score"] for c in range(2)]
            selected_idx = np.argmax(scores)

            for c in range(2):
                with columns[c + 1]:
                    suffix = "· selected: ✓" if c == selected_idx else ""
                    st.markdown("score: {:.3f} {}".format(results[r][c]["score"], suffix))
                    st.image(get_image_explanation(results[r][c]))

        st.markdown("---")


if __name__ == "__main__":
    main()
