import math
from sklearn.manifold import MDS, TSNE, SpectralEmbedding, Isomap
from sklearn.datasets import make_swiss_roll
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np


def get_cars_data():
    return pd.read_csv("cars.csv", header=None, index_col=0)


def get_swiss_roll_data():
    return make_swiss_roll()[0]


def get_nations_data() -> pd.DataFrame:
    # https://people.sc.fsu.edu/~jburkardt/datasets/mds/nations.dat
    return pd.read_csv("nations.csv", index_col="country")


def visualize_embedding(embeddings, data, title, add_labels):
    rows = int(len(embeddings) ** 0.5)
    columns = math.ceil(len(embeddings) / rows)
    fix, axs = plt.subplots(rows, columns, figsize=(12, 8))
    fix.suptitle(title)
    for i, (name, embedding) in enumerate(embeddings):
        row, col = i // rows, i % columns
        x_transformed = embedding.fit_transform(data)
        colors = cm.rainbow(np.linspace(0, 1, len(x_transformed)))
        axs[row, col].scatter(x_transformed[:, 0], x_transformed[:, 1], color=colors)
        axs[row, col].set_title(name)
        if add_labels:
            for row_data, (x, y) in zip(data.T, x_transformed):
                axs[row, col].text(x, y, f"{row_data}", )
    plt.show()


if __name__ == '__main__':
    data_sets = [
        ("Cars", get_cars_data(), True),
        ("Roll data", get_swiss_roll_data(), False),
        ("Nations", get_nations_data(), True),
    ]
    for name, data_set, add_labels in data_sets:
        iso = Isomap(n_components=2, n_neighbors=5)
        mds = MDS(n_components=2)
        tsne = TSNE(n_components=2)
        se = SpectralEmbedding(n_components=2)
        visualize_embedding(
            [("iso", iso), ("mds", mds), ("t-sne", tsne), ("se", se)],
            data_set, name, add_labels
        )
