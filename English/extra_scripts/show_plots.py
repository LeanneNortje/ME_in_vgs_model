from pathlib import Path
import numpy as np
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

im_emb_fn = Path(f'results/embeddings/filtered_image_embeddings.tsv')
im_lab_fn = Path(f'results/embeddings/filtered_image_labels.tsv')
aud_emb_fn = Path(f'results/embeddings/filtered_audio_embeddings.tsv')
aud_lab_fn = Path(f'results/embeddings/filtered_audio_labels.tsv')

marker_options = ['*', 'o', '^', 'd', 'X']

embeddings = []
with open(aud_emb_fn, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    for row in csvreader:
        a = np.expand_dims(np.asarray(row), axis=0)
        embeddings.append(a)

embeddings = np.concatenate(embeddings, axis=0)
print(embeddings.shape)

labels = []
tags = []
colors = {}
counts = {}
marker_count = 0
marker_dict = {}
filtered_embeddings = []
with open(aud_lab_fn, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    for n, row in enumerate(csvreader):
        if n != 0:

            if row[0] not in counts: counts[row[0]] = 0
            if counts[row[0]] < 50:

                filtered_embeddings.append(np.expand_dims(embeddings[n-1, :], axis=0))

                labels.append(row[0])
                tags.append(row[1])

                if row[0] not in colors: 
                    if row[1] == 'unseen': colors[row[0]] = 'green'
                    else: colors[row[0]] = 'blue'

                if row[0] not in marker_dict: 
                    marker_dict[row[0]] = marker_options[marker_count]
                    marker_count += 1

                counts[row[0]] += 1

filtered_embeddings = np.concatenate(filtered_embeddings, axis=0)

pca = PCA(n_components=2)
values = pca.fit_transform(filtered_embeddings)
for l in list(set(labels)):
    indices = np.where(np.asarray(labels) == l)[0]
    plt.scatter(values[indices, 0], values[indices, 1], c=colors[l], marker=marker_dict[l], label=l)
plt.legend()
plt.axis('off')
plt.savefig(f'results/audio_embedding_space.pdf',bbox_inches='tight')


embeddings = []
with open(im_emb_fn, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    for row in csvreader:
        a = np.expand_dims(np.asarray(row), axis=0)
        embeddings.append(a)

embeddings = np.concatenate(embeddings, axis=0)

labels = []
tags = []
counts = {}
filtered_embeddings = []
with open(im_lab_fn, 'r') as file:
    csvreader = csv.reader(file, delimiter='\t')
    for n, row in enumerate(csvreader):
        if n != 0:

            if row[0] not in counts: counts[row[0]] = 0
            if counts[row[0]] < 50:

                filtered_embeddings.append(np.expand_dims(embeddings[n-1, :], axis=0))

                labels.append(row[0])
                tags.append(row[1])
                if row[0] not in colors: 
                    if row[1] == 'unseen': colors[row[0]] = 'green'
                    else: colors[row[0]] = 'blue'

                if row[0] not in marker_dict: 
                    marker_dict[row[0]] = marker_options[marker_count]
                    marker_count += 1

                counts[row[0]] += 1

filtered_embeddings = np.concatenate(filtered_embeddings, axis=0)

pca = PCA(n_components=2)
values = pca.fit_transform(filtered_embeddings)
for l in list(set(labels)):
    indices = np.where(np.asarray(labels) == l)[0]
    plt.scatter(values[indices, 0], values[indices, 1], c=colors[l], marker=marker_dict[l], label=l)
plt.legend()
plt.axis('off')
plt.savefig(f'results/image_embedding_space.pdf',bbox_inches='tight')