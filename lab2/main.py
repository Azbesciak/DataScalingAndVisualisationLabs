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
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.008, color="blue", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0]/2 + 0.25, v[1]/2, "v{0}".format(i), color="blue")

        # Plot transformed vector.
        tv = A.dot(v)
        plt.quiver(0.0, 0.0, tv[0], tv[1], width=0.005, color="magenta", scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(tv[0] / 2 + 0.25, tv[1] / 2, "v{0}'".format(i), color="magenta")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.margins(0.05)
    # Plot eigenvectors
    plot_eigenvectors(A)
    plt.show()


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plt.quiver(0.0, 0.0, v[0], v[1], width=0.006, color=color, scale_units='xy', angles='xy', scale=1,
                   zorder=4)
        plt.text(v[0] / 2 + 0.25, v[1] / 2, "eigv{0}".format(i), color=color)


def plot_eigenvectors(A):
    """Plots all eigenvectors of the given 2x2 matrix A."""
    eig = np.linalg.eig(A)
    visualize_vectors(eig[1].T)


def EVD_decomposition(A):
    # TODO: Zad. 4.2. UzupeĹnij funkcjÄ tak by obliczaĹa rozkĹad EVD zgodnie z zadaniem.
    pass


def plot_attractors(A, vectors):
    # TODO: Zad. 4.3. UzupeĹnij funkcjÄ tak by generowaĹa wykres z atraktorami.
    pass


def test_A1(vectors):
    """Standard scaling transformation."""
    A = np.array([[2, 0],
                  [0, 2]])
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


def test_A2(vectors):
    A = np.array([[-1, 2],
                  [2, 1]])
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


def test_A3(vectors):
    A = np.array([[3, 1],
                  [0, 2]])
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)


def test_A4(vectors):
    A = np.array([[2, -1],
                  [1, 4]])
    EVD_decomposition(A)
    visualize_transformation(A, vectors)
    plot_attractors(A, vectors)



if __name__ == "__main__":
    vectors = vectors_uniform(k=8)
    test_A1(vectors)
    test_A2(vectors)
    test_A3(vectors)
    test_A4(vectors)