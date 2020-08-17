import numpy as np

EPS = 1e-3


def xy2mrx_v0(a_xy, b_pv, grid):
    """
    Encode xy to activations on a grid
    Activation symmetric so that flipping xy about
    b's v yields same values
    Clip activations that would be outside grid to map to last coordinate
    :param a_xy: n, 2 | (x_x, x_y)
    :param b_pv: n, 4 | (x_x, x_y, v_x, v_y)
    :param grid: (l, m) | symmetric (about 0) grid in 1D
        l | upper bound for coordinates
        m | number of grid points
        grid points are regularly spaced in [0, l]
    :return: n, m | encoding for each (of n) agent
    """
    n = a_xy.shape[0]
    m = int(grid[-1])
    mrx = np.zeros((n, m), dtype=np.float)
    pos_dif = a_xy - b_pv[:, :2]  # n, 2
    pos_dif_dot_v = np.einsum('ij, ij -> i', pos_dif, b_pv[:, 2:])  # n
    v_normsq = (b_pv[:, 2:] ** 2).sum(axis=1)
    v_normsq[v_normsq < EPS] = EPS
    perp_ds = (pos_dif ** 2).sum(axis=1) - (pos_dif_dot_v ** 2) / v_normsq
    perp_ds = np.sqrt(perp_ds)
    np.clip(perp_ds, a_min=None, a_max=grid[0]-EPS, out=perp_ds)
    d = grid[0] / (grid[1] - 1)
    for i in range(n):
        a, r = np.divmod(perp_ds[i], d)
        th = 1 - r/d
        inds = np.array([a, a+1], dtype=np.int)
        mrx[i, inds] = np.array([th, 1-th])
    return mrx


def xy2mrx_v1(a_xy, b_pv, grid):
    """
    Encode xy to activations on a grid
    Activation symmetric so that flipping xy about
    b's v yields same values
    Clip activations that would be outside grid to map to last coordinate
    :param a_xy: n, 2 | (x_x, x_y)
    :param b_pv: n, 4 | (x_x, x_y, v_x, v_y)
    :param grid: (l, m) | symmetric (about 0) grid in 1D
        l | upper bound for coordinates
        m | number of grid points
        grid points are regularly spaced in [0, l]
    :return: n, m | encoding for each (of n) agent
    """
    n = a_xy.shape[0]
    m = int(grid[-1])
    mrx = np.zeros((n, m), dtype=np.float)
    pos_dif = a_xy - b_pv[:, :2]  # n, 2
    pos_dif_dot_v = np.einsum('ij, ij -> i', pos_dif, b_pv[:, 2:])  # n
    v_normsq = (b_pv[:, 2:] ** 2).sum(axis=1)
    v_normsq[v_normsq < EPS] = EPS
    perp_ds = (pos_dif ** 2).sum(axis=1) - (pos_dif_dot_v ** 2) / v_normsq
    perp_ds = np.sqrt(perp_ds)
    np.clip(perp_ds, a_min=None, a_max=grid[0]-EPS, out=perp_ds)
    d = grid[0] / (grid[1] - 1)
    a, r = np.divmod(perp_ds, d)
    a = a.astype(np.int)
    th = 1 - r/d
    row_inds = np.arange(n)
    mrx[row_inds, a] = th
    mrx[row_inds, a + 1] = 1 - th
    return mrx


def evaluate_v_v0(a_pv, b_pv, grid, u, q, b_inds):
    """
    Evaluate velocity as
    \hat{v} = vq + v(1-q)[mrx_{b_ind}(x).dot(u)]
    :param a_pv: n, 4
    :param b_pv: k, 4
    :param grid: (l, m) | 2-tuple
    :param u: m, | weights to apply to grid encodings
    :param q: n, | q=0 for using grid encoding, q=1 for not
    :param b_inds: n, | index of b \in [0, k-1]
        (undefined for i st. q[i]=1)
    :return: n, 2 | \hat{v} for each (of n) agent
    """
    n = a_pv.shape[0]
    v_hat = a_pv[:, 2:].copy()
    d = grid[0] / (grid[1] - 1)
    for i in range(n):
        if q[i]:
            continue
        b_pv_i = b_pv[b_inds[i], :]
        pos_dif = a_pv[i, :2] - b_pv_i[:2]  # 2,
        pos_dif_dot_v = pos_dif.dot(b_pv_i[2:])
        v_normsq = (b_pv_i[2:] ** 2).sum(axis=-1)
        v_normsq = EPS if v_normsq < EPS else v_normsq
        perp_ds = (pos_dif ** 2).sum(axis=-1) - (pos_dif_dot_v ** 2) / v_normsq
        perp_ds = np.sqrt(perp_ds)
        perp_ds = np.clip(perp_ds, a_min=None, a_max=grid[0] - EPS)
        a, r = np.divmod(perp_ds, d)
        a = a.astype(np.int)
        th = 1 - r / d
        scaling = u[a] * th + u[a + 1] * (1 - th)
        v_hat[i, :] *= scaling
    return v_hat


def evaluate_v_v1(a_pv, b_pv, grid, u, q, b_inds):
    """
    Evaluate velocity as
    \hat{v} = vq + v(1-q)[mrx_{b_ind}(x).dot(u)]
    :param a_pv: n, 4
    :param b_pv: k, 4
    :param grid: (l, m) | 2-tuple
    :param u: m, | weights to apply to grid encodings
    :param q: n, | q=0 for using grid encoding, q=1 for not
    :param b_inds: n, | index of b \in [0, k-1]
    :return: n, 2 | \hat{v} for each (of n) agent
    """
    n = a_pv.shape[0]
    v_hat = a_pv[:, 2:].copy()
    d = grid[0] / (grid[1] - 1)

    # n->k pe dist
    pos_dif = a_pv[:, np.newaxis, :2] - b_pv[np.newaxis, :, :2]
    # n, k
    pos_dif_dot_v = np.einsum('ijk, jk -> ij', pos_dif, b_pv[:, 2:])
    # k
    v_normsq = (b_pv[:, 2:] ** 2).sum(axis=1)
    v_normsq[v_normsq < EPS] = EPS
    # n, k
    pe_dist_sq = (pos_dif ** 2).sum(axis=-1) - (pos_dif_dot_v ** 2) / v_normsq
    pe_dist = np.sqrt(pe_dist_sq)
    np.clip(pe_dist, a_min=None, a_max=grid[0] - EPS, out=pe_dist)

    # subset n, st. q=0: n_q0,
    q0_mask = q == 0
    # n_q0,
    a, r = np.divmod(pe_dist[q0_mask, b_inds[q0_mask]], d)
    a = a.astype(np.int)
    th = 1 - r / d
    scaling = u[a] * th + u[a + 1] * (1 - th)
    v_hat[q0_mask, :] = (v_hat[q0_mask, :].T * scaling).T
    return v_hat


def evaluate_v_particles_v0(a_pv, b_pv, grid, u, q, b_inds):
    """
    Evaluate velocity as
    \hat{v} = vq + v(1-q)[mrx_{b_ind}(x).dot(u)]
    :param a_pv: n_p, n, 4
    :param b_pv: k, 4
    :param grid: (l, m) | 2-tuple
    :param u: m, | weights to apply to grid encodings
    :param q: n_p, n | q=0 for using grid encoding, q=1 for not
    :param b_inds: n_p, n | index of b \in [0, k-1]
        (undefined for i st. q[i]=1)
    :return: n_p, n, 2 | \hat{v} for each (of n) agent
    """
    n_p, n = a_pv.shape[:2]
    v_hat = np.empty((n_p, n, 2))
    for i in range(n_p):
        v_hat[i, ...] = evaluate_v_v1(a_pv[i], b_pv, grid, u, q[i], b_inds[i])
    return v_hat


def evaluate_v_particles_v1(a_pv, b_pv, grid, u, q, b_inds):
    """
    Evaluate velocity as
    \hat{v} = vq + v(1-q)[mrx_{b_ind}(x).dot(u)]
    :param a_pv: n_p, n, 4
    :param b_pv: k, 4
    :param grid: (l, m) | 2-tuple
    :param u: m, | weights to apply to grid encodings
    :param q: n_p, n | q=0 for using grid encoding, q=1 for not
    :param b_inds: n_p, n | index of b \in [0, k-1]
        (undefined for i st. q[i]=1)
    :return: n_p, n, 2 | \hat{v} for each (of n) agent
    """
    n_p, n = a_pv.shape[:2]
    v_hat = evaluate_v_v1(
        a_pv.reshape(-1, 4), b_pv, grid, u, q.reshape(-1), b_inds.reshape(-1)).reshape(n_p, n, 2)
    return v_hat


def main_evaluate_v_particles():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    # seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n_p = 100
    n = 20
    k = 3
    a_pv = np.random.randn(n_p, n, 4)
    b_pv = np.random.randn(k, 4) * 2
    grid = np.array([5., 6])  # [0, 1, ..., 5]
    u = np.arange(grid[1]) / grid[1]
    q = np.random.randn(n_p, n) > 0
    b_inds = np.random.choice(k, n_p*n).reshape(n_p, n)
    print('---------------')

    x_true = evaluate_v_particles_v0(a_pv, b_pv, grid, u, q, b_inds)
    x_hat = evaluate_v_particles_v1(a_pv, b_pv, grid, u, q, b_inds)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))

    n_tries = 2
    args = (a_pv, b_pv, grid, u, q, b_inds)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=evaluate_v_particles_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=evaluate_v_particles_v1, args=args))/n_tries)


def main_evaluate_v():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    # seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n = 200
    k = 3
    a_pv = np.random.randn(n, 4)
    b_pv = np.random.randn(k, 4) * 2
    grid = np.array([5., 6])  # [0, 1, ..., 5]
    u = np.arange(grid[1]) / grid[1]
    q = np.random.randn(n) > 0
    b_inds = np.random.choice(k, n)
    print('---------------')

    x_true = evaluate_v_v0(a_pv, b_pv, grid, u, q, b_inds)
    x_hat = evaluate_v_v1(a_pv, b_pv, grid, u, q, b_inds)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))

    n_tries = 2
    args = (a_pv, b_pv, grid, u, q, b_inds)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=evaluate_v_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=evaluate_v_v1, args=args))/n_tries)


def main():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n = 300
    a_xy = np.random.randn(n, 2) * 5
    b_pv = np.random.randn(n, 4) * 2
    grid = np.array([5., 6])  # [0, 1, ..., 5]
    print('---------------')

    x_true = xy2mrx_v0(a_xy, b_pv, grid)
    x_hat = xy2mrx_v1(a_xy, b_pv, grid)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))

    n_tries = 2
    print(timeit('f(a, b, c)', number=n_tries, globals=dict(f=xy2mrx_v0, a=a_xy, b=b_pv, c=grid))/n_tries)
    print(timeit('f(a, b, c)', number=n_tries, globals=dict(f=xy2mrx_v1, a=a_xy, b=b_pv, c=grid))/n_tries)


if __name__ == '__main__':
    # main()
    main_evaluate_v()
    # main_evaluate_v_particles()
