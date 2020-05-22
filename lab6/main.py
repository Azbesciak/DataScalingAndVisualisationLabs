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
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def read_input_image(file_path: str):
    image = mpig.imread(file_path)
    if image.dtype == np.uint8:
        image = image / 255
    return image


def recreate_image(U: np.array, s: np.array, V: np.array, k: int = None):
    return np.dot(U[:, :k], np.dot(s[:k, :k], V[:k, :]))


def scikit_svg(image: np.array, k: int) -> np.array:
    U, s, V = svd(image, full_matrices=False)
    return recreate_image(U, np.diag(s), V, k)


def pseudo_reverse(mat: np.array) -> np.array:
    min_val = np.min(mat.shape)
    trimmed = mat[:min_val, :min_val]
    diagonal = np.diag(trimmed)
    inverse = np.where(diagonal != 0, 1 / diagonal, 0)
    res = np.diag(inverse)
    return res


def evd(image: np.array) -> np.array:
    eig, mat = np.linalg.eigh(image)
    sorted_indices = eig.argsort()[::-1]
    eig = eig[sorted_indices]
    mat = mat[:, sorted_indices]
    L = np.diag(eig)
    K = mat
    Kinv = np.linalg.inv(K)
    return np.real(K), np.real(L), np.real(Kinv)


def custom_svg(image: np.array, k: int) -> np.array:
    height, width = image.shape
    rotated = height > width
    if rotated:
        height, width = width, height
        image = image.T

    R = image @ image.T
    U, L_u, _ = evd(R)
    s = prepare_sigma(L_u, width, height)
    V_t = prepare_v_t(U, s, image, height, width)
    recreated = recreate_image(U, s, V_t, k)
    return recreated.T if rotated else recreated


def prepare_sigma(L_u, width, height):
    temp = np.sqrt(L_u[:, :min(width, height)])
    s = np.zeros((height, width), dtype=np.float64)
    s[:temp.shape[0], :temp.shape[1]] = temp
    return s


def prepare_v_t(U, s, image, height, width):
    s_inv = pseudo_reverse(s)
    V_t = np.zeros((width, width))
    V_t[:height, :] = s_inv @ U.T @ image
    return V_t


def compress(image: np.array, method: str, k: int or None):
    original_shape = image.shape
    if len(original_shape) == 3:
        image = image.reshape((original_shape[0], original_shape[1] * original_shape[2]))
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
