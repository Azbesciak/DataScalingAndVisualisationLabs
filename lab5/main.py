import math

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, SpectralEmbedding, Isomap, LocallyLinearEmbedding
from sklearn.datasets import make_swiss_roll
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np


def get_cars_data():
    return pd.read_csv("data/cars.csv", header=None, index_col=0)


def get_swiss_roll_data():
    swiss_roll, order = make_swiss_roll(n_samples=1000)
    ordered_indices = order.argsort()
    return swiss_roll[ordered_indices]


def get_nations_data() -> pd.DataFrame:
    # https://people.sc.fsu.edu/~jburkardt/datasets/mds/nations.dat
    return pd.read_csv("data/nations.csv", index_col="country")


def visualize_embedding(embeddings, data, title: str, add_labels, random_state):
    rows = int(len(embeddings) ** 0.5)
    columns = math.ceil(len(embeddings) / rows)
    fig, axs = plt.subplots(rows, columns, figsize=(20, 8))
    for i, (name, embedding) in enumerate(embeddings):
        row, col = i // columns, i % columns
        x_transformed = embedding.fit_transform(data)
        colors = cm.rainbow(np.linspace(0, 1, len(x_transformed)))
        axs[row, col].scatter(x_transformed[:, 0], x_transformed[:, 1], color=colors)
        axs[row, col].set_title(name)
        if add_labels:
            for row_data, (x, y) in zip(data.T, x_transformed):
                axs[row, col].text(x, y, f"{row_data}")

    fig.suptitle(f"{title} (random state = {random_state})", fontsize=24)
    plt.savefig(f"img/{title.lower().replace(' ', '_')}_{random_state}.png") # smaller size than SVG
    plt.show()


if __name__ == '__main__':
    data_sets = [
        ("Cars", get_cars_data(), True),
        ("Swiss Roll", get_swiss_roll_data(), False),
        ("Nations", get_nations_data(), True),
    ]
    for random_state in [1, 1020]:
        for name, data_set, add_labels in data_sets:
            iso3 = Isomap(n_components=2, n_neighbors=3)
            iso5 = Isomap(n_components=2, n_neighbors=5)
            iso7 = Isomap(n_components=2, n_neighbors=7)
            pca = PCA(n_components=2, random_state=random_state)
            mds = MDS(n_components=2, random_state=random_state)
            tsne = TSNE(n_components=2, random_state=random_state)
            se = SpectralEmbedding(n_components=2, random_state=random_state)
            lle = LocallyLinearEmbedding(n_components=2, random_state=random_state, n_neighbors=5)
            visualize_embedding(
                [("Isomap (n = 3)", iso3), ("Isomap (n = 5)", iso5), ("Isomap (n = 7)", iso7), ("Spectral Embedding", se),
                 ("PCA", pca), ("MDS scikit", mds), ("t-SNE", tsne), ("LocallyLinearEmbedding (n = 5)", lle)],
                data_set, name, add_labels, random_state
            )
