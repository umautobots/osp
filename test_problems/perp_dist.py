import numpy as np

EPS = 1e-5


def signed_pe_dist_v0(a_pv, b_pv):
    """
    :param a_pv: n, 4 | (x_x, x_y, v_x, v_y)
    :param b_pv: k, 4 | (x_x, x_y, v_x, v_y)
    :return:
        pe_dist: n, k | perpendicular distance of each a to each b
        pa_dist: n, k | parallel distance (signed)
    """
    n = a_pv.shape[0]
    k = b_pv.shape[0]
    pe_dist = np.zeros((n, k), dtype=np.float)
    pa_dist = np.zeros((n, k), dtype=np.float)
    for i in range(n):
        pos_dif = a_pv[i, :2] - b_pv[:, :2]
        pos_dif_dot_v = np.einsum('ij, ij -> i', pos_dif, b_pv[:, 2:])  # k
        v_normsq = (b_pv[:, 2:] ** 2).sum(axis=1)
        v_normsq[v_normsq < EPS] = EPS
        pa_dist[i, :] = pos_dif_dot_v/np.sqrt(v_normsq)
        pe_vec = pos_dif - (pa_dist[i, :]/np.sqrt(v_normsq) * b_pv[:, 2:].T).T
        pe_vec_dot_av = (pe_vec * a_pv[i, 2:]).sum(axis=1)
        pe_dif = (pos_dif**2).sum(axis=1) - pa_dist[i, :]**2
        pe_dif[pe_dif < EPS] = EPS
        pe_dist[i, :] = pe_dif
        pe_dist[i, :] = np.sqrt(pe_dist[i, :]) * np.sign(pe_vec_dot_av)
    return pe_dist, pa_dist


def signed_pe_dist_v1(a_pv, b_pv):
    """
    :param a_pv: n, 4 | (x_x, x_y, v_x, v_y)
    :param b_pv: k, 4 | (x_x, x_y, v_x, v_y)
    :return:
        pe_dist: n, k | perpendicular distance of each a to each b
        pa_dist: n, k | parallel distance (signed)
    """
    # n, k, 2
    pos_dif = a_pv[:, np.newaxis, :2] - b_pv[np.newaxis, :, :2]
    # n, k
    pos_dif_dot_v = np.einsum('ijk, jk -> ij', pos_dif, b_pv[:, 2:])
    # k
    v_normsq = (b_pv[:, 2:] ** 2).sum(axis=1)
    v_normsq[v_normsq < EPS] = EPS
    v_norm = np.sqrt(v_normsq)
    # n, k
    pa_dist = pos_dif_dot_v / v_norm
    # (n, k, 2) - (n, k) * (k, 2)
    pe_vec = pos_dif - (pa_dist / v_norm)[..., np.newaxis] * b_pv[:, 2:][np.newaxis, ...]
    # n, k
    pe_vec_dot_av = np.einsum('ijk, ik -> ij', pe_vec, a_pv[:, 2:])
    pe_dist = (pos_dif ** 2).sum(axis=-1) - pa_dist ** 2
    pe_dist[pe_dist < EPS] = EPS
    pe_dist = np.sqrt(pe_dist) * np.sign(pe_vec_dot_av)
    return pe_dist, pa_dist


def signed_pe_dist_frames_v0(a_pv, b_pv):
    """
    :param a_pv: n_frames, n, 4 | (x_x, x_y, v_x, v_y)
    :param b_pv: n_frames, k, 4 | (x_x, x_y, v_x, v_y)
    :return:
        pe_dist: n_frames, n, k | perpendicular distance of each a to each b
        pa_dist: n_frames, n, k | parallel distance (signed)
    """
    n_frames, n = a_pv.shape[:2]
    k = b_pv.shape[1]
    pe_dist = np.empty((n_frames, n, k))
    pa_dist = np.empty((n_frames, n, k))
    for i in range(n_frames):
        pe_dist[i], pa_dist[i] = signed_pe_dist_v1(a_pv[i], b_pv[i])
    return pe_dist, pa_dist


def signed_pe_dist_frames_v1(a_pv, b_pv):
    """
    :param a_pv: n_frames, n, 4 | (x_x, x_y, v_x, v_y)
    :param b_pv: n_frames, k, 4 | (x_x, x_y, v_x, v_y)
    :return:
        pe_dist: n_frames, n, k | perpendicular distance of each a to each b
        pa_dist: n_frames, n, k | parallel distance (signed)
    """
    # n_frames, n, k, 2
    pos_dif = a_pv[:, :, np.newaxis, :2] - b_pv[:, np.newaxis, :, :2]
    # n_frames, n, k
    pos_dif_dot_v = np.einsum('lijk, ljk -> lij', pos_dif, b_pv[..., 2:])
    # n_frames, k
    v_normsq = (b_pv[..., 2:] ** 2).sum(axis=-1)
    v_normsq[v_normsq < EPS] = EPS
    v_norm = np.sqrt(v_normsq)
    # n_frames, n, k
    pa_dist = pos_dif_dot_v / v_norm[:, np.newaxis, :]
    # n_frames, n, k, 2
    pe_vec = pos_dif - (pa_dist / v_norm[:, np.newaxis, :])[..., np.newaxis] *\
             b_pv[:, np.newaxis, :, 2:]
    # n_frames, n, k
    pe_vec_dot_av = np.einsum('lijk, lik -> lij', pe_vec, a_pv[..., 2:])
    pe_dist = (pos_dif ** 2).sum(axis=-1) - pa_dist ** 2
    pe_dist = np.sqrt(pe_dist) * np.sign(pe_vec_dot_av)
    return pe_dist, pa_dist


def signed_pe_dist_particles_v0(a_pv, b_pv):
    """
    :param a_pv: n_p, n, 4 | (x_x, x_y, v_x, v_y)
    :param b_pv: k, 4 | (x_x, x_y, v_x, v_y)
    :return:
        pe_dist: n_p, n, k | perpendicular distance of each a to each b
        pa_dist: n_p, n, k | parallel distance (signed)
    """
    n_p, n = a_pv.shape[:2]
    k = b_pv.shape[0]
    pe_dist = np.empty((n_p, n, k))
    pa_dist = np.empty((n_p, n, k))
    for i in range(n_p):
        pe_dist[i], pa_dist[i] = signed_pe_dist_v1(a_pv[i], b_pv)
    return pe_dist, pa_dist


def signed_pe_dist_particles_v1(a_pv, b_pv):
    """
    :param a_pv: n_p, n, 4 | (x_x, x_y, v_x, v_y)
    :param b_pv: k, 4 | (x_x, x_y, v_x, v_y)
    :return:
        pe_dist: n_p, n, k | perpendicular distance of each a to each b
        pa_dist: n_p, n, k | parallel distance (signed)
    """
    # n_p, n, k, 2
    pos_dif = a_pv[:, :, np.newaxis, :2] - b_pv[np.newaxis, np.newaxis, :, :2]
    # n_p, n, k
    pos_dif_dot_v = np.einsum('lijk, jk -> lij', pos_dif, b_pv[:, 2:])
    # k
    v_normsq = (b_pv[:, 2:] ** 2).sum(axis=-1)
    v_normsq[v_normsq < EPS] = EPS
    v_norm = np.sqrt(v_normsq)
    # n_p, n, k
    pa_dist = pos_dif_dot_v / v_norm
    # n_p, n, k, 2
    pe_vec = pos_dif - (pa_dist / v_norm)[..., np.newaxis] *\
        b_pv[np.newaxis, np.newaxis, :, 2:]
    # n_p, n, k
    pe_vec_dot_av = np.einsum('lijk, lik -> lij', pe_vec, a_pv[..., 2:])
    pe_dist = (pos_dif ** 2).sum(axis=-1) - pa_dist ** 2
    pe_dist[pe_dist < EPS] = EPS
    pe_dist = np.sqrt(pe_dist) * np.sign(pe_vec_dot_av)
    return pe_dist, pa_dist


def pe_dist_v0(a_xy, b_pv):
    """
    :param a_xy: n, 2 | (x_x, x_y)
    :param b_pv: k, 4 | (x_x, x_y, v_x, v_y)
    :return:
        pe_dist: n, k | perpendicular distance of each a to each b
        pa_dist: n, k | parallel distance (signed)
    """
    n = a_xy.shape[0]
    k = b_pv.shape[0]
    pe_dist = np.zeros((n, k), dtype=np.float)
    pa_dist = np.zeros((n, k), dtype=np.float)
    for i in range(n):
        pos_dif = a_xy[i] - b_pv[:, :2]
        pos_dif_dot_v = np.einsum('ij, ij -> i', pos_dif, b_pv[:, 2:])  # k
        v_normsq = (b_pv[:, 2:] ** 2).sum(axis=1)
        v_normsq[v_normsq < EPS] = EPS
        pa_dist[i, :] = pos_dif_dot_v/np.sqrt(v_normsq)
        pe_dist[i, :] = (pos_dif ** 2).sum(axis=1) - pa_dist[i, :] ** 2
        pe_dist[i, :] = np.sqrt(pe_dist[i, :])
    return pe_dist, pa_dist


def pe_dist_v1(a_xy, b_pv):
    """
    :param a_xy: n, 2 | (x_x, x_y)
    :param b_pv: k, 4 | (x_x, x_y, v_x, v_y)
    :return:
        pe_dist: n, k | perpendicular distance of each a to each b
        pa_dist: n, k | parallel distance (signed)
    """
    # n, k, 2
    pos_dif = a_xy[:, np.newaxis, :2] - b_pv[np.newaxis, :, :2]
    # n, k
    pos_dif_dot_v = np.einsum('ijk, jk -> ij', pos_dif, b_pv[:, 2:])
    # k
    v_normsq = (b_pv[:, 2:] ** 2).sum(axis=1)
    v_normsq[v_normsq < EPS] = EPS
    # n, k
    pa_dist_sq = pos_dif_dot_v / np.sqrt(v_normsq)
    pe_dist_sq = (pos_dif ** 2).sum(axis=-1) - pa_dist_sq ** 2
    return np.sqrt(pe_dist_sq), pa_dist_sq


def main_signed_pe_dist_particles():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    # seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n_p = 100
    n = 30
    k = 3
    a_pv = np.random.randn(n_p, n, 4)
    b_pv = np.random.randn(k, 4) * 2
    print('---------------')

    x_true = signed_pe_dist_particles_v0(a_pv, b_pv)
    x_hat = signed_pe_dist_particles_v1(a_pv, b_pv)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[0] - x_hat[0])))
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[1] - x_hat[1])))

    n_tries = 2
    args = (a_pv, b_pv)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=signed_pe_dist_particles_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=signed_pe_dist_particles_v1, args=args))/n_tries)


def main_signed_pe_dist_frames():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    # seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n_frames = 30
    n = 30
    k = 3
    a_pv = np.random.randn(n_frames, n, 4)
    b_pv = np.random.randn(n_frames, k, 4) * 2
    print('---------------')

    x_true = signed_pe_dist_frames_v0(a_pv, b_pv)
    x_hat = signed_pe_dist_frames_v1(a_pv, b_pv)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[0] - x_hat[0])))
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[1] - x_hat[1])))

    n_tries = 2
    args = (a_pv, b_pv)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=signed_pe_dist_frames_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=signed_pe_dist_frames_v1, args=args))/n_tries)


def main_signed_pe_dist():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    # seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n = 30
    k = 3
    a_pv = np.random.randn(n, 4)
    b_pv = np.random.randn(k, 4) * 2
    print('---------------')

    x_true = signed_pe_dist_v0(a_pv, b_pv)
    x_hat = signed_pe_dist_v1(a_pv, b_pv)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[0] - x_hat[0])))
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[1] - x_hat[1])))

    n_tries = 2
    args = (a_pv, b_pv)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=signed_pe_dist_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=signed_pe_dist_v1, args=args))/n_tries)


def main_pe_dist():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    # seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n = 300
    k = 2
    a_xy = np.random.randn(n, 2)
    b_pv = np.random.randn(k, 4) * 2
    print('---------------')

    x_true = pe_dist_v0(a_xy, b_pv)
    x_hat = pe_dist_v1(a_xy, b_pv)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[0] - x_hat[0])))
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[1] - x_hat[1])))

    n_tries = 2
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=pe_dist_v0, a=a_xy, b=b_pv))/n_tries)
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=pe_dist_v1, a=a_xy, b=b_pv))/n_tries)


def main():
    a_pv = np.array([[5, 0, 0, -1.]])
    b_pv = np.array([[0., 5, 2, 0], [0., -5, 2, 0]])
    signed_pe_dist_v0(a_pv, b_pv)
    print(signed_pe_dist_v0(a_pv, b_pv))

    a_pv = np.array([[0, 0, 1, 0.]])
    b_pv = np.array([[0., 5, 2, -.1]])
    signed_pe_dist_v0(a_pv, b_pv)
    print(signed_pe_dist_v0(a_pv, b_pv))


if __name__ == '__main__':
    # main()
    # main_pe_dist()
    # main_signed_pe_dist()
    # main_signed_pe_dist_frames()
    main_signed_pe_dist_particles()
