"""
Inference as implemented assumes are agents continuously observed (unlike fitting)
Sample by MC, using velocity estimated by optimizing over r, q during observations
Sample velocity, then to determine weights
0) global opt over r, q: n_vic^n_obs * 2^n_obs choices
1) global opt over r, q: but use independence, so n_obs * (n_vic * 2) choices
2) greedy over r, q: pick best r, then pick best q: n_obs
Here use (1)
"""
import numpy as np
from baselines import velocity_model
from scipy import special as sp
from test_problems import risk_distances as rd
from test_problems import grid_encoding as rt_grid
from test_problems import u_grid_encoding as u_grid
from test_problems import sample_rq
from test_problems import sample_driftless_rw
from test_problems import perp_dist
from test_problems import opt_out_rq as opt
from ss_model import predict_utils
from time import time

N_SAMPLES = 100
EPS = 1e-5


def predict(ped_xy, vic_xy, dataset_id, datafile_id,
            n_steps=0, parameters=None, **kwargs):
    """

    :param ped_xy: n_obs, n_ped, 2
    :param vic_xy: n_obs, n_vic, 2
    :param dataset_id:
    :param datafile_id:
    :param n_steps:
    :param parameters:
    :param kwargs:
    :return:
        ped_xy_hat: n_steps, n_ped, 2, n_samples
        p: n_ped, n_samples
        data_dict:
    """
    t0 = time()
    if vic_xy.size == 0:
        return velocity_model.predict_constant_velocity(
            ped_xy, vic_xy, dataset_id, datafile_id, n_steps=n_steps, **kwargs)
    n_obs = ped_xy.shape[0]
    vic_pv = predict_utils.make_vic_pv(vic_xy[:n_obs], n_steps, parameters.dt)
    vic_pv = predict_utils.rm_stationary(vic_pv)
    ped_xy_hat, p = predict_mc(ped_xy, vic_pv, n_steps, parameters, N_SAMPLES)
    td = time() - t0
    return np.moveaxis(ped_xy_hat, 0, -1), p.T, {'duration': td}


def predict_given_vic(ped_xy, vic_xy, dataset_id, datafile_id,
                      n_steps=0, parameters=None, **kwargs):
    """

    :param ped_xy: n_obs, n_ped, 2
    :param vic_xy: n_obs + n_steps, n_vic, 2
    :param dataset_id:
    :param datafile_id:
    :param n_steps:
    :param parameters:
    :param kwargs:
    :return:
        ped_xy_hat: n_steps, n_ped, 2, n_samples
        p: n_ped, n_samples
        data_dict:
    """
    if vic_xy.size == 0:
        return velocity_model.predict_constant_velocity(
            ped_xy, vic_xy, dataset_id, datafile_id, n_steps=n_steps, **kwargs)
    vic_pv = predict_utils.make_vic_pv_given(vic_xy, parameters.dt)
    vic_pv = predict_utils.rm_stationary(vic_pv)
    ped_xy_hat, p = predict_mc(ped_xy, vic_pv, n_steps, parameters, N_SAMPLES)
    return np.moveaxis(ped_xy_hat, 0, -1), p.T, {}


def predict_mc(ped_xy, vic_pv, n_steps, parameters, n_samples):
    """
    Pure MC, no resampling
    state = [x v r q] \in \reals^4 \times Z \times {0, 1}
    :param ped_xy: n_obs, n_ped, 2
    :param vic_pv: n_obs + n_steps, n_vic, 4 | [x_x x_y v_x v_y]
    :param n_steps:
    :param parameters:
    :param n_samples:
    :return:
        preds: n_samples, n_steps, n_ped, 2
        p: n_samples, n_ped
    """
    n_obs, n_ped = ped_xy.shape[:2]
    state = np.empty((n_samples, n_obs + n_steps, n_ped, 6), dtype=np.float)
    state[:, :n_obs, ...], nll = sample_obs_states(
        ped_xy, vic_pv[:n_obs], parameters, n_samples)

    for i in range(n_obs, n_obs + n_steps):
        state[:, i, ...] = sample_next(state[:, i-1, ...], vic_pv[i, ...], parameters)

    preds = state[:, n_obs:, :, :2]
    # preds = state[:, :, :, :2]
    p = sp.softmax(-nll, axis=0)
    return preds, p


def sample_obs_states(ped_xy, vic_pv, parameters, n_samples):
    """
    Sample initial x, v for n_obs states
    - use observed x
    - sample v
    - calculate nll by optimizing over r, q
    :param ped_xy: n_obs, n_ped, 2
    :param vic_pv: n_obs, n_vic, 4
    :param parameters:
    :param n_samples:
    :return:
        xvvh: n_samples, n_obs, n_ped, 6 | [x v vh]
        nll: n_samples, n_ped
    """
    n_obs, n_ped = ped_xy.shape[:2]
    ped_x_vh = np.empty((n_obs, n_ped, 4))
    ped_x_vh[..., :2] = ped_xy
    ped_x_vh[:-1, :, 2:] = (ped_xy[1:] - ped_xy[:-1]) / parameters.dt
    ped_x_vh[-1, :, 2:] = ped_x_vh[-2, :, 2:]
    state = np.empty((n_samples, n_obs, n_ped, 6), dtype=np.float)
    state[..., :2] = ped_xy
    state[..., 4:6] = ped_x_vh[:, :, 2:]
    state[..., 2:4], nll_v = sample_v(ped_x_vh, vic_pv, parameters, n_samples)
    nll = np.empty((n_samples, n_obs, n_ped), dtype=np.float)
    for i in range(n_obs):
        nll[:, i, :] = opt.optimize_out_softmax_rq_particles_v1(
            state[:, i, ...], vic_pv[i], parameters.rt_grid, parameters.beta,
            parameters.mrx_grid, parameters.u, parameters.sigma_x, parameters.dt)[2]
    return state, nll.sum(axis=1) + nll_v


def sample_next(state, vic_pv, parameters):
    """
    For pedestrians that can interact with no vic - ignore
    For remaining, mask no-interaction vic and sample r, q
    :param state: n_samples, n_ped, 6 | [x v r q]
    :param vic_pv: n_vic, 4 | [x v]
    :param parameters:
    :return:
    """
    n_samples, n_ped = state.shape[:2]
    state_next = np.empty_like(state)
    pe_dist, pa_dist = perp_dist.signed_pe_dist_particles_v1(
        state[..., :4], vic_pv)
    # n_samples, n_ped, n_vic
    is_ignore = predict_utils.is_ignore(pe_dist, pa_dist, parameters.mrx_grid[0])
    ignore_n_mask = np.all(is_ignore, axis=-1)
    n_ni = (~ignore_n_mask).sum()
    state_next[ignore_n_mask, :2] = state[ignore_n_mask, :2] +\
        state[ignore_n_mask, 2:4] * parameters.dt
    state_next[ignore_n_mask, 2:4] = state[ignore_n_mask, 2:4] + \
        np.random.randn(n_samples*n_ped - n_ni, 2) * parameters.sigma_v
    if n_ni == 0:
        # ignore all vic
        return state_next
    # remaining ped have at least 1 interacting vic
    # n_ni, n_vic, 2
    rt_all = rd.dist2rt_v1(state[~ignore_n_mask, :4], vic_pv)
    rt_all[rt_all < EPS] = EPS
    rt_all = np.log10(rt_all)
    rt_all[..., 0] /= 2
    # n_ni, n_vic, m
    z_all = rt_grid.rt2enc_v1(rt_all, parameters.rt_grid)
    # n_ni, n_vic
    z_all_dot_beta = z_all.dot(-parameters.beta[1:]) - parameters.beta[0]
    # n_ni,
    r_s, q_s = sample_rq.sample_softmax_rq_v0(
        z_all_dot_beta, is_ignore[~ignore_n_mask])
    v_hat = u_grid.evaluate_v_v1(
        state[~ignore_n_mask, :4], vic_pv, parameters.mrx_grid, parameters.u,
        q_s == 1, r_s.astype(np.int))
    state[~ignore_n_mask, 4] = r_s
    state[~ignore_n_mask, 5] = q_s
    state_next[~ignore_n_mask, :2] = state[~ignore_n_mask, :2] + v_hat * parameters.dt
    state_next[~ignore_n_mask, 2:4] = state[~ignore_n_mask, 2:4] +\
        np.random.randn(n_ni, 2) * parameters.sigma_v
    return state_next


def sample_v(ped_pv, vic_pv, parameters, n_samples):
    """
    Observed x: no interacting vic, must have q=1
    Case some observed consecutive [x_t, x_{t+1}]:
    - use difference, v_t

    :param ped_pv: n_obs, n_ped, 4 | [x v] with v from differences
    :param vic_pv: n_obs, n_vic, 4
    :param parameters:
    :param n_samples:
    :return:
        v_s: n_samples, n_obs, n_ped, 2
        nll: n_samples, n_ped
    """
    n_obs, n_ped = ped_pv.shape[:2]
    # n_obs, n_ped, n_vic
    pe_dist, pa_dist = perp_dist.signed_pe_dist_frames_v1(ped_pv, vic_pv)
    is_ignore = predict_utils.is_ignore(pe_dist, pa_dist, parameters.mrx_grid[0])
    # n_obs-1, n_ped
    is_v_obs = is_ignore.all(axis=-1)
    is_v_obs = is_v_obs[1:] & is_v_obs[:-1]
    # n_ped
    is_any_ped_obs = is_v_obs.any(axis=0)
    v_s = np.empty((n_samples, n_obs, n_ped, 2))
    nll = np.zeros((n_samples, n_ped))
    if is_any_ped_obs.any():
        # _sig = 0*np.sqrt(2)*parameters.sigma_x/parameters.dt
        _sig = np.sqrt(2)*parameters.sigma_x/parameters.dt
        v = np.empty((is_any_ped_obs.sum(), n_obs, 2))
        v.fill(np.nan)
        v[:, :-1, :] = np.swapaxes(ped_pv[:-1, is_any_ped_obs, 2:], 0, 1)
        # v_s_obs, nll_obs = sample_driftless_rw.sample_n_v0(
        #     v, n_samples, _sig, parameters.sigma_v)
        # nll[:, is_any_ped_obs] = nll_obs.T
        v_s_obs = sample_driftless_rw.sample_n_exact_v0(
            v, n_samples, _sig, parameters.sigma_v)
        # n_ped, n_samples, n_obs, 2 -> n_samples, n_obs, n_ped, 2
        v_s[:, :, is_any_ped_obs, :] = np.transpose(v_s_obs, (1, 2, 0, 3))
    v_s[:, :, ~is_any_ped_obs, :] = sample_unobs_v(
        ped_pv[:, ~is_any_ped_obs, :], n_samples)
    return v_s, nll


def sample_unobs_v(ped_pv, n_samples):
    """

    :param ped_pv: n_obs, n_ped, 4 | [x v] with v from differences
    :param n_samples:
    :return:
        v_s: n_samples, n_obs, n_ped, 2
    """
    n_obs, n_ped = ped_pv.shape[:2]
    v_mean = ped_pv[:-1, :, 2:].mean(axis=0)
    v_mean = 1.4 * (v_mean.T / np.linalg.norm(v_mean, axis=-1)).T
    rv = (np.random.rand(n_samples, n_ped, 2) - 0.5)/5 * 2.
    v_s = rv[:, np.newaxis, ...] + v_mean[np.newaxis, np.newaxis, :, :]
    return v_s
