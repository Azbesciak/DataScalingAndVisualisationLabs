#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


def vectors_uniform(k):
    """Uniformly generates k vectors."""
    vectors = []
    for a in np.linspace(0, 2 * np.pi, k, endpoint=False):
        vectors.append(2 * np.array([np.sin(a), np.cos(a)]))
    return vectors


def visualize_transformation(A, vectors):
    """Plots original and transformed vectors for a given 2x2 transformation matrix A and a list of 2D vectors."""
    for i, v in enumerate(vectors):
        # Plot original vector.
        plot_arrow(i, v, color="blue", width=0.008, prefix="v")

        # Plot transformed vector.
        tv = A.dot(v)
        plot_arrow(i, tv, color="magenta", width=0.005, prefix="v")
    set_margins()
    plot_eigenvectors(A)
    plt.show()


def set_margins():
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)


def plot_arrow(i, v, color: str, width: float, prefix: str):
    plt.quiver(0.0, 0.0, v[0], v[1], width=width, color=color, scale_units='xy', angles='xy', scale=1, zorder=4)
    plt.text(v[0] / 2 + 0.25, v[1] / 2, f"{prefix}{i}", color=color)


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plot_arrow(i, v, width=0.006, color=color, prefix="eigv")


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    _, eig = np.linalg.eig(A)
    visualize_vectors(eig.T)


def EVD_decomposition(A):
    """
    https://stackoverflow.com/questions/50487118/eigendecomposition-makes-me-wonder-in-numpy
    """
    eig, mat = np.linalg.eig(A)
    L = np.diag(eig)
    K = mat
    Kinv = np.linalg.inv(K)
    EVD = K @ L @ Kinv
    print(f"{'-' * 60}\nA:   {A.tolist()}\nEVD: {EVD.tolist()}\n\n" +
          "K-1: {Kinv.tolist()}\nL:   {L.tolist()}\nK:   {K.tolist()}")
    # assert np.allclose(EVD, A), f"EVD: {EVD.tolist()}, A: {A.tolist()}"


def plot_attractors(A, vectors):
    # TODO: Zad. 4.3. UzupeĹnij funkcjÄ tak by generowaĹa wykres z atraktorami.
    pass


def show_eigen_info(matrix: list, vectors):
    A = np.array(matrix)
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


if __name__ == "__main__":
    vectors = vectors_uniform(k=8)
    show_eigen_info([[2, 0], [0, 2]], vectors)
    show_eigen_info([[-1, 2], [2, 1]], vectors)
    show_eigen_info([[3, 1], [0, 2]], vectors)
    # Here assertion fails. (?)
    show_eigen_info([[2, -1], [1, 4]], vectors)
