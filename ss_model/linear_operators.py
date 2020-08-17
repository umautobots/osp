import numpy as np
import scipy.sparse.linalg as la
from scipy import sparse


def forward_dif(n, dtype=np.float64):
    """
    Simulates forward difference matrix C \in (n-1, n)
    C = [-1 1         = I_n[1:, :] - I_n[:-1,:]
           -1 1
             ...
               -1 1]
    :param n:
    :param dtype:
    :return:
    """
    class Op(la.LinearOperator):
        def __init__(self, *args, **kwargs):
            super(Op, self).__init__(*args, **kwargs)

        def _matmat(self, X):
            return X[1:, :] - X[:-1, :]

        def _rmatvec(self, z_):
            z = z_.reshape(-1,)
            y = np.hstack([-z[0], z[1:]-z[:-1], z[-1]])
            if z_.ndim == 2:
                y = y.reshape(-1, 1)
            return y

        def _rmatmat(self, Z):
            Y = np.vstack((-Z[[0], :], Z[1:, :]-Z[:-1, :], Z[[-1], :]))
            return Y
    return Op(dtype, (n-1, n))


def left_mult_mat(A, T, dtype=np.float64):
    """
    Apply matrix A to matrix of data:
    x = vec(X) with X \in (n, T)
    Y = AX
    y = vec(Y)
    for operator: x -> y
    :param A: m, n
    :param T:
    :param dtype:
    :return:
    """
    m, n = A.shape

    class Op(la.LinearOperator):
        def __init__(self, *args, **kwargs):
            super(Op, self).__init__(*args, **kwargs)

        def _matvec(self, x):
            X = x.reshape(n, -1, order='F')
            Y = A.dot(X)
            y = Y.reshape(-1, order='F')
            if x.ndim == 2:
                y = y.reshape(-1, 1)
            return y

        def _rmatvec(self, y):
            Y = y.reshape(m, -1, order='F')
            X = A.T.dot(Y)
            x = X.reshape(-1, order='F')
            if y.ndim == 2:
                x = x.reshape(-1, 1)
            return x
    return Op(dtype, (m*T, n*T))


def select_inds(n, inds, dtype=np.float64):
    """
    Simulate matrix S \in (k, n) that selects k indices
    ex:
    I_n[:k, :]: select the first k values in order
    :param n: size of vector being hit
    :param inds: S_{ij} = 1 iff inds[i] = j
    :param dtype:
    :return:
    """
    class Op(la.LinearOperator):
        def __init__(self, *args, **kwargs):
            super(Op, self).__init__(*args, **kwargs)

        def _matmat(self, X):
            return X[inds, :]

        def _rmatvec(self, y_):
            y = y_.reshape(-1,)
            x = np.zeros((n,), dtype=dtype)
            x[inds] = y
            if y_.ndim == 2:
                x = x.reshape(-1, 1)
            return x

        def _rmatmat(self, Y):
            X = np.zeros((n, Y.shape[1]), dtype=dtype)
            X[inds, :] = Y
            return X
    k = len(inds)
    return Op(dtype, (k, n))


def block_diag(A_list, dtype=np.float64):
    """
    Make block-diagonal linear operator L from list [A, B, ...]
    L = [ A
            B
              ... ]
    :param A_list: list of A_i \in (m_i, n_i)
    :param dtype:
    :return:
    """
    k = len(A_list)
    n = np.cumsum([0] + [A_list[i].shape[1] for i in range(k)])
    m = np.cumsum([0] + [A_list[i].shape[0] for i in range(k)])

    class Op(la.LinearOperator):
        def __init__(self, *args, **kwargs):
            super(Op, self).__init__(*args, **kwargs)

        def _matmat(self, X):
            return np.vstack([A_list[i].dot(X[n[i]:n[i + 1], :])
                              for i in range(k)])

        def _rmatvec(self, y_):
            Y = y_.reshape(-1, 1)
            x = self._rmatmat(Y)
            if y_.ndim == 2:
                x = x.reshape(-1, 1)
            return x

        def _rmatmat(self, Y):
            return np.vstack([A_list[i].T.dot(Y[m[i]:m[i + 1], :])
                              for i in range(k)])
    return Op(dtype, (m[-1], n[-1]))


def vstack(A_list, dtype=np.float64):
    """
    Stack linear operators vertically
    :param A_list: each A_i \in (m_i, n)
    :param dtype:
    :return:
    """
    k = len(A_list)
    m = np.cumsum([0] + [Ai.shape[0] for Ai in A_list])
    n = A_list[0].shape[1]

    class Op(la.LinearOperator):
        def __init__(self, *args, **kwargs):
            super(Op, self).__init__(*args, **kwargs)

        def _matmat(self, X):
            return np.vstack([Ai.dot(X) for Ai in A_list])

        def _rmatvec(self, y_):
            Y = y_.reshape(-1, 1)
            x = self._rmatmat(Y)
            if y_.ndim == 2:
                x = x.reshape(-1, 1)
            return x

        def _rmatmat(self, Y):
            return np.sum([A_list[i].T.dot(Y[m[i]:m[i+1], :]) for i in range(k)], axis=0)
    return Op(dtype, (m[-1], n))


def _dense_forward_dif(n):
    i_n = np.eye(n)
    return i_n[1:, :] - i_n[:-1, :]


def _dense_left_mult_mat(A, T):
    block_list = [[A if i == t else 0*A for i in range(T)]
                  for t in range(T)]
    return np.block(block_list)


def _dense_select_inds(n, inds):
    k = len(inds)
    A = np.zeros((k, n))
    A[np.arange(k), inds] = 1
    return A


def _dense_block_diag(A_list):
    A = sparse.block_diag(A_list, format='csr')
    return A


def _dense_vstack(A_list):
    return np.vstack(A_list)


def _check_mult_transpose_mult(x, A, B):
    z_a = A.dot(x)
    z_b = B.dot(x)
    assert np.isclose(z_a, z_b).all()
    assert np.isclose(A.T.dot(z_a), B.T.dot(z_b)).all()
    X = np.array([x, x+2, x+5]).T
    Z_a = A.dot(X)
    Z_b = B.dot(X)
    assert np.isclose(Z_a, Z_b).all()
    assert np.isclose(A.T.dot(Z_a), B.T.dot(Z_b)).all()


def main_check_forward_dif():
    n = 10
    x = np.arange(n, dtype=np.float)
    C_dense = _dense_forward_dif(n)
    C_op = forward_dif(n)
    _check_mult_transpose_mult(x, C_dense, C_op)


def main_check_left_mult_mat():
    seed = 0
    np.random.seed(seed)
    m, n = 4, 3
    T = 10
    A = np.arange(m*n, dtype=np.float).reshape(m, n)
    x = np.random.randn(n, T).reshape(-1, order='F')
    C_dense = _dense_left_mult_mat(A, T)
    C_op = left_mult_mat(A, T)
    _check_mult_transpose_mult(x, C_dense, C_op)


def main_check_select_inds():
    n = 10
    inds = np.arange(n)[::2]
    x = np.arange(n, dtype=np.float)
    C_dense = _dense_select_inds(n, inds)
    C_op = select_inds(n, inds)
    _check_mult_transpose_mult(x, C_dense, C_op)


def main_check_block_diag():
    seed = 0
    np.random.seed(seed)
    k = 10
    mn = [(np.random.randint(1, 6, 2)) for _ in range(k)]
    A_list = [np.random.randn(*mn[i]) for i in range(k)]
    x = np.random.randn(sum([mn[i][1] for i in range(k)]))
    C_dense = _dense_block_diag(A_list)
    C_op = block_diag(A_list)
    _check_mult_transpose_mult(x, C_dense, C_op)


def main_check_vstack():
    seed = 0
    np.random.seed(seed)
    k = 10
    n = 5
    mn = [(np.random.randint(1, 6), n) for _ in range(k)]
    A_list = [np.random.randn(*mni) for mni in mn]
    x = np.random.randn(n)
    C_dense = _dense_vstack(A_list)
    C_op = vstack(A_list)
    _check_mult_transpose_mult(x, C_dense, C_op)


if __name__ == '__main__':
    main_check_forward_dif()
    main_check_left_mult_mat()
    main_check_select_inds()
    main_check_block_diag()
    main_check_vstack()
