
import torch


def transform_closure(X_bin):
    """
    Convert binary relation matrix to permutation matrix
    :param X_bin: torch.tensor which is binarized by a threshold
    :return:
    """
    temp = torch.zeros_like ( X_bin )
    N = X_bin.shape[0]
    for k in range ( N ):
        for i in range ( N ):
            for j in range ( N ):
                temp[i][j] = X_bin[i, j] or (X_bin[i, k] and X_bin[k, j])
    vis = torch.zeros ( N )
    match_mat = torch.zeros_like ( X_bin )
    for i, row in enumerate ( temp ):
        if vis[i]:
            continue
        for j, is_relative in enumerate ( row ):
            if is_relative:
                vis[j] = 1
                match_mat[j, i] = 1
    return match_mat


if __name__ == '__main__':
    test_X = torch.randn ( 20, 20 ) > 0
    transform_closure ( test_X )
