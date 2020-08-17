import numpy as np

EPS = 1e-3
M = 100


def dist2rt_v0(a_pv, b_pv):
    """

    :param a_pv: n, 4 | (x_x, x_y, v_x, v_y)
    :param b_pv: k, 4
    :return: (n, k, 2)
        [d^2, tau] for each ped in 'a' to each vic in 'b'
    """
    n, k = a_pv.shape[0], b_pv.shape[0]
    rt = np.zeros((n, k, 2), dtype=np.float)
    for i in range(n):
        for j in range(k):
            # calculate t*
            v_dif = a_pv[i, 2:] - b_pv[j, 2:]
            p_dif = a_pv[i, :2] - b_pv[j, :2]
            vd_dot_pd = v_dif.dot(p_dif)
            vd_normsq = max(v_dif.dot(v_dif), EPS)
            rt[i, j, 1] = -vd_dot_pd/vd_normsq

            # calculate min distance
            pd_normsq = p_dif.dot(p_dif)
            rt[i, j, 0] = pd_normsq - vd_dot_pd**2/vd_normsq
    mask = rt[:, :, 1] < 0
    rt[mask, 1] = M
    rt[mask, 0] = M**2
    return rt


def dist2rt_v1(a_pv, b_pv):
    """

    :param a_pv: n, 4 | (x_x, x_y, v_x, v_y)
    :param b_pv: k, 4
    :return: (n, k, 2)
        [d^2, tau] for each ped in 'a' to each vic in 'b'
    """
    n, k = a_pv.shape[0], b_pv.shape[0]
    rt = np.zeros((n, k, 2), dtype=np.float)

    # n, k, 2
    p_dif = a_pv[:, np.newaxis, :2] - b_pv[np.newaxis, :, :2]
    v_dif = a_pv[:, np.newaxis, 2:] - b_pv[np.newaxis, :, 2:]
    # n, k
    vd_dot_pd = np.einsum('ijk, ijk -> ij', p_dif, v_dif)
    vd_normsq = (v_dif ** 2).sum(axis=2)
    vd_normsq[vd_normsq < EPS] = EPS
    rt[:, :, 1] = -vd_dot_pd / vd_normsq

    pd_normsq = (p_dif ** 2).sum(axis=2)
    rt[:, :, 0] = pd_normsq - vd_dot_pd**2/vd_normsq

    mask = rt[:, :, 1] < 0
    rt[mask, 1] = M
    rt[mask, 0] = M ** 2
    return rt


def dist2rt_particles_v0(a_pv, b_pv):
    """

    :param a_pv: n_p, n, 4 | (x_x, x_y, v_x, v_y), with n_p particles
    :param b_pv: k, 4
    :return: (n_p, n, k, 2)
        [d^2, tau] for each ped in 'a' to each vic in 'b'
    """
    n_p, n, k = a_pv.shape[0], a_pv.shape[1], b_pv.shape[0]
    rt = np.zeros((n_p, n, k, 2), dtype=np.float)
    for i in range(n_p):
        rt[i, ...] = dist2rt_v1(a_pv[i, ...], b_pv)
    return rt


def dist2rt_particles_v1(a_pv, b_pv):
    """

    :param a_pv: n_p, n, 4 | (x_x, x_y, v_x, v_y), with n_p particles
    :param b_pv: k, 4
    :return: (n_p, n, k, 2)
        [d^2, tau] for each ped in 'a' to each vic in 'b'
    """
    n_p, n, k = a_pv.shape[0], a_pv.shape[1], b_pv.shape[0]
    rt = np.zeros((n_p, n, k, 2), dtype=np.float)

    # n_p, n, k, 2
    p_dif = a_pv[:, :, np.newaxis, :2] - b_pv[np.newaxis, :, :2]
    v_dif = a_pv[:, :, np.newaxis, 2:] - b_pv[np.newaxis, :, 2:]
    # n_p, n, k
    vd_dot_pd = np.einsum('lijk, lijk -> lij', p_dif, v_dif)
    vd_normsq = (v_dif ** 2).sum(axis=3)
    vd_normsq[vd_normsq < EPS] = EPS
    rt[:, :, :, 1] = -vd_dot_pd / vd_normsq

    pd_normsq = (p_dif ** 2).sum(axis=3)
    rt[:, :, :, 0] = pd_normsq + vd_dot_pd * rt[:, :, :, 1]

    mask = rt[:, :, :, 1] < 0
    rt[mask, 1] = M
    rt[mask, 0] = M ** 2
    return rt


def main_particles():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n_p = 100
    n = 30
    k = 2
    ped_pv = np.random.rand(n_p, n, 4)
    vic_pv = np.random.rand(k, 4)
    ped_pv[:, :, :2] *= 10
    vic_pv[:, :2] *= 10
    print('---------------')

    x_true = dist2rt_particles_v0(ped_pv, vic_pv)
    x_hat = dist2rt_particles_v1(ped_pv, vic_pv)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))

    n_tries = 10
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=dist2rt_particles_v0, a=ped_pv, b=vic_pv))/n_tries)
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=dist2rt_particles_v1, a=ped_pv, b=vic_pv))/n_tries)


def main():
    from time import time
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n = 30
    k = 2
    ped_pv = np.random.rand(n, 4)
    vic_pv = np.random.rand(k, 4)
    ped_pv[:, :2] *= 10
    vic_pv[:, :2] *= 10

    # print(ped_pv)
    # print(vic_pv)
    print('---------------')

    x_true = dist2rt_v0(ped_pv, vic_pv)
    # print(x_true)
    t0 = time()
    x_hat = dist2rt_v0(ped_pv, vic_pv)
    print('time elapsed: {:0.5f}'.format(time() - t0))
    # print(x_hat)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))

    n_tries = 10
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=dist2rt_v0, a=ped_pv, b=vic_pv))/n_tries)
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=dist2rt_v1, a=ped_pv, b=vic_pv))/n_tries)


if __name__ == '__main__':
    # main()
    main_particles()

