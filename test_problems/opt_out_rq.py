import numpy as np
from scipy import special as sp
import test_problems.perp_dist as perp
import test_problems.risk_distances as rd
from test_problems import grid_encoding as b_grid
import test_problems.u_grid_encoding as u_grid


R_IGNORED = -1
EPS = 1e-5


def optimize_out_softmax_rq_v0(a_pvvh, b_pv, rt_grid, beta,
                               mrx_grid, u, sigma_x, dt):
    """
    Calculate r, q, nll for a single frame (n=n_ped, k=n_vic)
    Ignore k that are outside the range of grid, setting:
    - r = R_IGNORED
    - q = 1
    :param a_pvvh: n, 6 | [x v \hat{v}], v is given, \hat{v} from differences
    :param b_pv: k, 4
    :param rt_grid: (lx, ly, dx, dy, nx, ny) | 6-tuple
    :param beta: 1+nx*ny, | weights for rt_grid encoding
    :param mrx_grid: (l, m) | 2-tuple
    :param u: m, | weights for  mrx_grid encoding
    :param sigma_x:
    :param dt:
    :return: r, q that minimize total (r, q, x) nll given v
        r: n,
        q: n,
        nll: n, | loss for each agent
    """
    n = a_pvvh.shape[0]
    k = b_pv.shape[0]
    r = np.zeros(n, dtype=np.int) + R_IGNORED
    q = np.ones(n, dtype=np.int)
    nll = np.zeros(n)
    pe_dist, pa_dist = perp.signed_pe_dist_v1(a_pvvh[:, :4], b_pv)
    is_ignore = (pe_dist < -mrx_grid[0]) | (0 < pe_dist) | (pa_dist < 0)
    ignore_n_mask = np.all(is_ignore, axis=1)
    n_ni = (~ignore_n_mask).sum()

    # is_ignore -> compute only x_q1 loss
    scaling = 0.5 * (dt / sigma_x) ** 2
    nll[ignore_n_mask] = ((a_pvvh[ignore_n_mask, 2:4] -
                           a_pvvh[ignore_n_mask, 4:6])**2).sum(axis=-1) * scaling
    if n_ni == 0:
        return r, q, nll
    nll_all = np.zeros((n_ni, k, 2))
    # *some* k may be valid for *some* n but not for others
    ni_inds = np.arange(n)[~ignore_n_mask]
    for i, n_ind in enumerate(ni_inds):
        k_s = (~is_ignore[n_ind, :]).sum()
        if k_s == 0:
            # ignore all vic
            continue
        # make 'x'
        rt_all = rd.dist2rt_v1(a_pvvh[[n_ind], :4], b_pv[~is_ignore[n_ind, :], :])
        rt_all[rt_all < EPS] = EPS
        rt_all = np.log10(rt_all)
        rt_all[..., 0] /= 2
        # n=1, k_s, m
        z_all = b_grid.rt2enc_v1(rt_all, rt_grid)
        # k_s,
        z_all_dot_beta = (z_all.dot(-beta[1:]) - beta[0])[0]
        # k_s, : - log likelihood
        r_loss = -(z_all_dot_beta - sp.logsumexp(z_all_dot_beta))
        nll_all[i, ~is_ignore[n_ind, :], :] += np.broadcast_to(r_loss, (2, k_s)).T

        # k_s,
        q1_loss = np.logaddexp(0*z_all_dot_beta, z_all_dot_beta)
        q0_loss = -z_all_dot_beta + q1_loss
        nll_all[i, ~is_ignore[n_ind, :], 0] += q0_loss
        nll_all[i, ~is_ignore[n_ind, :], 1] += q1_loss

        # calculate v_slow
        # - replicate a_pv to calculate it for all k_s vic at once
        # k_s, 2
        v_slow = u_grid.evaluate_v_v1(
            np.broadcast_to(a_pvvh[[n_ind], :4], (k_s, 4)), b_pv[~is_ignore[n_ind, :], :],
            mrx_grid, u,
            np.zeros(k_s, dtype=np.int), np.arange(k_s, dtype=np.int)
        )
        x_q0_loss = ((v_slow - a_pvvh[n_ind, 4:6])**2).sum(axis=-1) * scaling
        nll_all[i, ~is_ignore[n_ind, :], 0] += x_q0_loss
        # 1,
        x_q1_loss = ((a_pvvh[n_ind, 2:4] - a_pvvh[n_ind, 4:6])**2).sum() * scaling
        nll_all[i, ~is_ignore[n_ind, :], 1] += x_q1_loss

    nll_all[is_ignore[ni_inds]] = np.inf
    # flattened inds wrt [0, k*2)
    min_inds = nll_all.reshape(n_ni, -1).argmin(axis=-1)
    r[ni_inds], q[ni_inds] = np.unravel_index(min_inds, (k, 2))
    nll_flat_inds = np.arange(0, n_ni*k*2, k*2) + min_inds
    nll[~ignore_n_mask] = nll_all.flat[nll_flat_inds]
    return r, q, nll


def optimize_out_softmax_rq_v1(a_pvvh, b_pv, rt_grid, beta,
                               mrx_grid, u, sigma_x, dt):
    """
    Calculate r, q, nll for a single frame (n=n_ped, k=n_vic)
    Ignore k that are outside the range of grid, setting:
    - r = R_IGNORED
    - q = 1
    :param a_pvvh: n, 6 | [x v \hat{v}], v is given, \hat{v} from differences
    :param b_pv: k, 4
    :param rt_grid: (lx, ly, dx, dy, nx, ny) | 6-tuple
    :param beta: 1+nx*ny, | weights for rt_grid encoding
    :param mrx_grid: (l, m) | 2-tuple
    :param u: m, | weights for  mrx_grid encoding
    :param sigma_x:
    :param dt:
    :return: r, q that minimize total (r, q, x) nll given v
        r: n,
        q: n,
        nll: n, | loss for each agent
    """
    n = a_pvvh.shape[0]
    k = b_pv.shape[0]
    r = np.zeros(n, dtype=np.int) + R_IGNORED
    q = np.ones(n, dtype=np.int)
    nll = np.zeros(n)
    pe_dist, pa_dist = perp.signed_pe_dist_v1(a_pvvh[:, :4], b_pv)
    is_ignore = (pe_dist < -mrx_grid[0]) | (0 < pe_dist) | (pa_dist < 0)
    ignore_n_mask = np.all(is_ignore, axis=1)
    n_ni = (~ignore_n_mask).sum()

    # is_ignore -> compute only x_q1 loss
    scaling = 0.5 * (dt / sigma_x) ** 2
    nll[ignore_n_mask] = ((a_pvvh[ignore_n_mask, 2:4] -
                           a_pvvh[ignore_n_mask, 4:6])**2).sum(axis=-1) * scaling
    if n_ni == 0:
        return r, q, nll
    nll_all = np.zeros((n_ni, k, 2))
    is_ignore_sub = is_ignore[~ignore_n_mask]
    # *some* k may be valid for *some* n but not for others
    # how to ignore bad k while still processing?
    # in logsumexp use b=is_ignore_sub.astype(int)
    ni_inds = np.arange(n)[~ignore_n_mask]
    a_pvvh_ni = a_pvvh[~ignore_n_mask]

    rt_all = rd.dist2rt_v1(a_pvvh_ni[:, :4], b_pv)
    rt_all[rt_all < EPS] = EPS
    rt_all = np.log10(rt_all)
    rt_all[..., 0] /= 2
    # n_ni, k, m
    z_all = b_grid.rt2enc_v1(rt_all, rt_grid)
    # n_ni, k
    z_all_dot_beta = z_all.dot(-beta[1:]) - beta[0]
    r_loss = -(z_all_dot_beta.T - sp.logsumexp(
        z_all_dot_beta, b=(~is_ignore_sub).astype(np.int), axis=-1)).T
    nll_all += r_loss[..., np.newaxis]

    q1_loss = np.logaddexp(0 * z_all_dot_beta, z_all_dot_beta)
    q0_loss = -z_all_dot_beta + q1_loss
    nll_all[..., 0] += q0_loss
    nll_all[..., 1] += q1_loss

    # n_ni*k, 2
    # a_pvvh_ni (n_ni, 4) -> (n_ni*k, 4)
    v_slow = u_grid.evaluate_v_v1(
        np.repeat(a_pvvh_ni[:, :4], k, axis=0), b_pv,
        mrx_grid, u,
        np.zeros(n_ni*k, dtype=np.int),
        np.broadcast_to(np.arange(k, dtype=np.int), (n_ni, k)).reshape(-1),
    )
    # n_ni, k
    x_q0_loss = ((v_slow.reshape(n_ni, k, 2) -
                  a_pvvh_ni[:, np.newaxis, 4:6]) ** 2).sum(axis=-1) * scaling
    nll_all[..., 0] += x_q0_loss

    # n_ni,
    x_q1_loss = ((a_pvvh_ni[:, 2:4] - a_pvvh_ni[:, 4:6]) ** 2).sum(axis=-1) * scaling
    nll_all[..., 1] += np.broadcast_to(x_q1_loss, (k, n_ni)).T

    nll_all[is_ignore_sub] = np.inf
    # flattened inds wrt [0, k*2)
    min_inds = nll_all.reshape(n_ni, -1).argmin(axis=-1)
    r[ni_inds], q[ni_inds] = np.unravel_index(min_inds, (k, 2))
    nll_flat_inds = np.arange(0, n_ni*k*2, k*2) + min_inds
    nll[~ignore_n_mask] = nll_all.flat[nll_flat_inds]
    return r, q, nll


def optimize_out_softmax_rq_particles_v0(
        a_pvvh, b_pv, rt_grid, beta, mrx_grid, u, sigma_x, dt):
    """
    Calculate r, q, nll for a single frame (n=n_ped, k=n_vic)
    Ignore k that are outside the range of grid, setting:
    - r = R_IGNORED
    - q = 1
    :param a_pvvh: n_p, n, 6 | [x v \hat{v}], v is given, \hat{v} from differences
    :param b_pv: k, 4
    :param rt_grid: (lx, ly, dx, dy, nx, ny) | 6-tuple
    :param beta: 1+nx*ny, | weights for rt_grid encoding
    :param mrx_grid: (l, m) | 2-tuple
    :param u: m, | weights for  mrx_grid encoding
    :param sigma_x:
    :param dt:
    :return: r, q that minimize total (r, q, x) nll given v
        r: n_p, n,
        q: n_p, n,
        nll: n_p, n, | loss for each agent
    """
    n_p, n = a_pvvh.shape[:2]
    r = np.empty((n_p, n), dtype=np.int)
    q = np.empty((n_p, n), dtype=np.int)
    nll = np.empty((n_p, n))
    for i in range(n_p):
        r[i], q[i], nll[i] = optimize_out_softmax_rq_v1(
            a_pvvh[i], b_pv, rt_grid, beta, mrx_grid, u, sigma_x, dt)
    return r, q, nll


def optimize_out_softmax_rq_particles_v1(
        a_pvvh, b_pv, rt_grid, beta, mrx_grid, u, sigma_x, dt):
    """
    Calculate r, q, nll for a single frame (n=n_ped, k=n_vic)
    Ignore k that are outside the range of grid, setting:
    - r = R_IGNORED
    - q = 1
    :param a_pvvh: n_p, n, 6 | [x v \hat{v}], v is given, \hat{v} from differences
    :param b_pv: k, 4
    :param rt_grid: (lx, ly, dx, dy, nx, ny) | 6-tuple
    :param beta: 1+nx*ny, | weights for rt_grid encoding
    :param mrx_grid: (l, m) | 2-tuple
    :param u: m, | weights for  mrx_grid encoding
    :param sigma_x:
    :param dt:
    :return: r, q that minimize total (r, q, x) nll given v
        r: n_p, n,
        q: n_p, n,
        nll: n_p, n, | loss for each agent
    """
    n_p, n = a_pvvh.shape[:2]
    r, q, nll = optimize_out_softmax_rq_v1(
        a_pvvh.reshape(-1, 6), b_pv, rt_grid, beta, mrx_grid, u, sigma_x, dt)
    return r.reshape(n_p, n), q.reshape(n_p, n), nll.reshape(n_p, n)


def main_optimize_out_softmax_rq_particles():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n_p = 100
    n = 20
    k = 3
    a_pvvh = np.random.randn(n_p, n, 6)
    b_pv = np.random.randn(k, 4) * 2
    args = (a_pvvh, b_pv) + make_parameters()
    print('---------------')

    x_true = optimize_out_softmax_rq_particles_v0(*args)
    x_hat = optimize_out_softmax_rq_particles_v1(*args)
    for i in range(len(x_true)):
        # print(x_true[i])
        print('diff: {:0.4f}'.format(np.linalg.norm(x_true[i] - x_hat[i])))
    # print(x_true[2])
    # print(x_hat[2])

    n_tries = 2
    print(timeit('f(*args)', number=n_tries, globals=dict(f=optimize_out_softmax_rq_particles_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=optimize_out_softmax_rq_particles_v1, args=args))/n_tries)



def main_optimize_out_softmax_rq():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n = 20
    k = 3
    a_pvvh = np.random.randn(n, 6)
    b_pv = np.random.randn(k, 4) * 2
    args = (a_pvvh, b_pv) + make_parameters()
    print('---------------')

    x_true = optimize_out_softmax_rq_v0(*args)
    x_hat = optimize_out_softmax_rq_v1(*args)
    for i in range(len(x_true)):
        # print(x_true[i])
        print('diff: {:0.4f}'.format(np.linalg.norm(x_true[i] - x_hat[i])))
    # print(x_true[2])
    # print(x_hat[2])

    n_tries = 2
    print(timeit('f(*args)', number=n_tries, globals=dict(f=optimize_out_softmax_rq_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=optimize_out_softmax_rq_v1, args=args))/n_tries)


def make_parameters():
    dt = 0.1
    sigma_x = 0.05
    mrx_grid = (6., 7)
    rt_grid = (0, 0, 0.4, 0.4, 5, 5)
    u = np.array([0., -1., 0.52, 0.23, 0.43, 0.89, 0.08])
    beta = np.hstack((
        6.23, np.array([
            [-19.65, -27.79, 21.83, 0.4, 0.02],
            [-8.03, -1.77, 3.35, 1.07, 0.07],
            [24.54, 5.11, 2.13, 1.83, 0.21],
            [0., 0.06, 0.74, 0.53, 0.04],
            [0.,   0.,   0.,   0., 1.54],
        ]).ravel()))
    return rt_grid, beta, mrx_grid, u, sigma_x, dt


if __name__ == '__main__':
    # main_optimize_out_softmax_rq()
    main_optimize_out_softmax_rq_particles()
