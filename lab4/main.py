#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from swd_labs.lab4.zadania_pca import data_example1, data_random, plot_pca_result_2d


def pca_sklearn(data, n_comp=None):
    """Reduces dimensionality using decomposition.PCA class from scikit-learn library.

    :param data: (np.array) Input data. Rows: observations, columns: features.
    :param n_comp: (int) Number of dimensions (components). By default equal to the number of features (variables).
    :return: (np.array) The transformed data after changing the coordinate system, possibly with removed dimensions with
    the smallest variance.
    """
    assert isinstance(data, np.ndarray)
    assert n_comp <= data.shape[1], "The number of components cannot be greater than the number of features"
    if n_comp is None:
        n_comp = data.shape[1]
    pca = PCA(n_comp)
    adjusted = centralize(data)
    return pca.fit_transform(adjusted)


def centralize(data):
    return data - data.mean(axis=0, keepdims=True)


def pca_manual(data, n_comp=None):
    """Reduces dimensionality by using your own PCA implementation.

    :param data: (np.array) Input data. Rows: observations, columns: features.
    :param n_comp: (int) Number of dimensions (components). By default equal to the number of features (variables).
    :return: (np.array) The transformed data after changing the coordinate system, possibly reduced.
    """
    assert isinstance(data, np.ndarray)
    assert n_comp <= data.shape[1], "The number of components cannot be greater than the number of features"
    if n_comp is None:
        n_comp = data.shape[1]

    adjusted = centralize(data)
    cov_mat = np.cov(adjusted, rowvar=False)
    print("\nCOVARIANCE MATRIX:")
    print(cov_mat)

    # vectors are already normalized as doc says
    e, eig = np.linalg.eig(cov_mat)
    sorted_indices = np.argsort(-e)
    e_sorted, eig_sorted = e[sorted_indices], eig.T[sorted_indices]
    print("\nSORTED EIGEN VALUES:")
    print(e_sorted)
    print("\nSORTED EIGEN VECTORS:")
    print(eig_sorted)

    selected_eig = eig_sorted[:n_comp].T
    transformed_adj_data = adjusted @ selected_eig
    cov_transformed_adj_data = np.cov(transformed_adj_data, rowvar=False)
    print("\nCOVARIANCE MATRIX OF THE TRANSFORMED DATA:")
    print(cov_transformed_adj_data)
    return transformed_adj_data


def run_pca_comparison(X, n_comp=1):
    Y1 = pca_manual(X, n_comp=n_comp)
    Y2 = pca_sklearn(X, n_comp=n_comp)
    print("\nDifference: {0}".format(abs((Y1 - Y2)).sum()))
    plot_pca_result_2d(X, Y1, Y2)


if __name__ == "__main__":
    data = data_example1()
    run_pca_comparison(data, n_comp=2)

    data = data_random(100, x_mean=0.0, x_std=1.0, y_mean=0.0, y_std=4.0, angle=-math.pi / 4)
    run_pca_comparison(data, n_comp=2)
