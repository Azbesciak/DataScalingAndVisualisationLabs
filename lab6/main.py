import argparse
import matplotlib.image as mpig
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

CUSTOM_SVD = "custom"
LIB_SVD = "library"
AVAILABLE_SVD = [CUSTOM_SVD, LIB_SVD]


def get_input_params():
    parser = argparse.ArgumentParser()
    parser.add_help = True
    parser.add_argument("-f", "--file", help="path to the image", required=True)
    parser.add_argument("-out", help="output image path")
    parser.add_argument("-svd", help="SVD implementation to use", choices=AVAILABLE_SVD, default=AVAILABLE_SVD[0])
    parser.add_argument("-k", help="number of singular values used for compression", type=int)
    return parser.parse_args()


def make_result_presentation(result, output_path):
    plt.axes([0, 0, 1, 1])
    plt.axis("off")
    plt.imshow(result)
    if output_path is None:
        plt.show()
    else:
        plt.savefig(params.out, bbox_inches='tight', pad_inches=0)


def read_input_image(file_path: str):
    image = mpig.imread(file_path)
    if image.dtype == np.uint8:
        image = image / 255
    return image


def scikit_svg(image: np.array, k: int) -> np.array:
    U, s, V = svd(image, full_matrices=False)
    return np.dot(U[:, :k], np.dot(np.diag(s[:k]), V[:k, :]))


def custom_svg(image: np.array, k: int) -> np.array:
    return image


def compress(image: np.array, method: str, k: int or None):
    original_shape = image.shape
    if len(original_shape) == 3:
        image = image.reshape((original_shape[0], original_shape[1] * 3))
    reconst_matrix = scikit_svg(image, k) if method == LIB_SVD else custom_svg(image, k)
    if len(original_shape) == 3:
        reconst_matrix = reconst_matrix.reshape(original_shape)
    return reconst_matrix


if __name__ == '__main__':
    params = get_input_params()
    if params.k is not None and params.k < 0:
        raise ValueError("k must be not negative or not defined")
    image = read_input_image(params.file)
    compressed_image = compress(image, params.svd, params.k)
    make_result_presentation(compressed_image, params.out)
