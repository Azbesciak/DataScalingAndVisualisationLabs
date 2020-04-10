#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


default_color = "#000000"
colors = [["#F44336", "#FF9800"], ["#4CAF50", "#009688"], ["#E91E63", "#F06292"]]
retries = 10
equ_eps = 1e-10


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
        plot_arrow(i, v, color="blue", width=0.008, text=f"v{i}")

        # Plot transformed vector.
        tv = A.dot(v)
        plot_arrow(i, tv, color="magenta", width=0.005, text=f"v{i}")
    set_margins(6)
    plot_eigenvectors(A)
    plt.show()


def set_margins(dim):
    plt.xlim([-dim, dim])
    plt.ylim([-dim, dim])
    plt.margins(0.05)


def plot_arrow(i, v, color: str, width: float, text: str = None, at_end=False, zorder=4):
    plt.quiver(0.0, 0.0, v[0], v[1], width=width, color=color, scale_units='xy', angles='xy', scale=1, zorder=zorder)
    plt.text(v[0] if at_end else v[0] / 2 + 0.25, v[1] if at_end else v[1] / 2, text, color=color, zorder=zorder)


def visualize_vectors(vectors, color="green"):
    """Plots all vectors in the list."""
    for i, v in enumerate(vectors):
        plot_arrow(i, v, width=0.006, color=color, text=f"eigv{i}")


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
          f"K-1: {Kinv.tolist()}\nL:   {L.tolist()}\nK:   {K.tolist()}")
    # assert np.allclose(EVD, A, equ_eps), f"EVD: {EVD.tolist()}, A: {A.tolist()}"




def plot_attractors(A, vectors):
    e, eig = np.linalg.eig(A)
    unique_eigen_vals, unique_vectors = get_unique_vectors_with_inverse(e, eig)
    eig_with_reversed = np.concatenate((unique_vectors, unique_vectors * -1))
    __plot_attractors(A, eig_with_reversed, unique_eigen_vals, unique_vectors, vectors)
    for i, v in enumerate(unique_vectors):
        palette = colors[min(i, len(colors) - 1)]
        plot_arrow(i, v, width=0.012, color=palette[0], text=e[i], at_end=True, zorder=10)
        plot_arrow(i, v * -1, width=0.012, color=palette[1], at_end=True, zorder=10)
    set_margins(2)
    plt.grid()
    plt.show()


def __plot_attractors(A, eig_with_reversed, unique_eigen_vals, unique_vectors, vectors):
    is_unit = np.allclose(np.identity(len(unique_eigen_vals)), unique_vectors)
    unique_vectors_count = len(unique_vectors)
    for i, v in enumerate(vectors):
        normalized = normalize(v)
        color = default_color
        if not is_unit:
            derived = normalized
            for r in range(retries):
                derived = A.dot(derived)
                derived = normalize(derived)
            dif = [np.sum(np.abs(derived - eigVec)) for eigVec in eig_with_reversed]
            argmax = np.argmin(dif)
            color = colors[argmax % unique_vectors_count][argmax // unique_vectors_count]
        # Plot original vector.
        plot_arrow(i, normalized, color=color, width=0.004)


def get_unique_vectors_with_inverse(e, eig):
    eigT = eig.T
    eigen_values = e.tolist()
    return np.array([v for i, v in enumerate(eigen_values) if is_unique(eigen_values, eigT, i)]), \
           np.array([v for i, v in enumerate(eigT) if is_unique(eigen_values, eigT, i)])


def is_unique(eigen_values: list, eigen_vectors: np.array, i: int):
    first_index_in_list = eigen_values.index(eigen_values[i])
    return i == first_index_in_list or not np.allclose(eigen_vectors[i], eigen_vectors[first_index_in_list], equ_eps)


def normalize(derived):
    return derived / max(np.sqrt(np.sum(derived ** 2)), 1e-10)


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
