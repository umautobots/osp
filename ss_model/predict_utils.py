import numpy as np


class Parameters:
    """
    Note:
        use -beta for sampling r, since q is fitted as 'q=1 <=> go'
    """

    def __init__(self):
        self.u = np.array([])
        self.beta = np.array([])
        self.sigma_x = 0.
        self.sigma_v = 0.
        self.dt = 0.
        self.mrx_grid = np.array([])
        self.rt_grid = np.array([])


def make_parameters_dut_train():
    parameters = Parameters()
    parameters.dt = 0.1
    parameters.sigma_x = 0.05
    parameters.sigma_v = 0.0035
    parameters.mrx_grid = (6., 7)
    parameters.rt_grid = (0, 0, 0.4, 0.4, 5, 5)
    parameters.u = np.array([0., -1., 0.52, 0.23, 0.43, 0.89, 0.08])
    parameters.beta = np.hstack((
        6.23, np.array([
            [-19.65, -27.79, 21.83, 0.4, 0.02],
            [-8.03, -1.77, 3.35, 1.07, 0.07],
            [24.54, 5.11, 2.13, 1.83, 0.21],
            [0., 0.06, 0.74, 0.53, 0.04],
            [0.,   0.,   0.,   0., 1.54],
        ]).ravel()))
    return parameters


def make_parameters_ind_train():
    parameters = Parameters()
    parameters.dt = 0.1
    parameters.sigma_x = 0.05
    parameters.sigma_v = 0.0027
    parameters.mrx_grid = (6., 7)
    parameters.rt_grid = (0, 0, 0.4, 0.4, 5, 5)
    parameters.u = np.array([-0.39, 1., 0., 0.34, 0.46, 0.7, 0.06])
    parameters.beta = np.hstack((
        5.62, np.array([
            [-16.47, -13.18, 0.59, 2.25, 0.48],
            [-5.36, -2.5, 0.56, 2.57, 0.94],
            [23.31, 2.79, 1.07, 2.44, 0.87],
            [-4.5, 3.33, 2.56, 1.58, 0.06],
            [-4.31, 1.17, 0.84, 0.45, 4.09],
        ]).ravel()))
    return parameters


def is_ignore(pe_dist, pa_dist, tau_dist):
    """
    :param pe_dist: nd-array
    :param pa_dist: nd-array
    :param tau_dist: max perpendicular distance of interactions (eg from u_grid)
    :return:
        mask: nd-array with same dimensions
    """
    mask = (pe_dist < -tau_dist) | (0 < pe_dist) | (pa_dist < 0)
    return mask


def rm_stationary(pv, tau_v=0.5):
    """
    Remove if less than threshold for all frames
    :param pv: n_frames, k, 4
    :param tau_v:
    :return:
    """
    is_stationary = np.linalg.norm(pv[..., 2:], axis=-1) < tau_v
    return pv[:, ~(is_stationary.all(axis=0)), :]


def make_vic_pv(vic_xy, n_steps, dt):
    """
    Fill in velocity using differencing, and assuming constant velocity
    :param vic_xy: n_obs, n_vic, 2 | n_obs > 1
    :param n_steps:
    :param dt:
    :return:
    """
    n_obs, n_vic = vic_xy.shape[:2]
    vic_pv = np.empty((n_obs + n_steps, n_vic, 4))
    vic_pv[:n_obs, :, :2] = vic_xy
    vic_pv[:n_obs-1, :, 2:] = (vic_xy[1:] - vic_xy[:-1]) / dt
    vic_pv[n_obs-1:, :, 2:] = vic_pv[n_obs-2, :, 2:]
    mean_vdt = vic_pv[n_obs-1, :, :2] - vic_pv[n_obs-2, :, :2]
    vic_pv[n_obs:, :, :2] = vic_pv[n_obs-1, :, :2] + \
        mean_vdt[np.newaxis, :, :] * np.arange(1, n_steps + 1)[:, np.newaxis, np.newaxis]
    return vic_pv


def make_vic_pv_given(vic_xy, dt):
    """
    :param vic_xy: n_obs + n_pred, n_vic, 2
    :param dt:
    :return:
    """
    n_frames, n_vic = vic_xy.shape[:2]
    vic_pv = np.empty((n_frames, n_vic, 4))
    vic_pv[:, :, :2] = vic_xy
    vic_pv[:-1, :, 2:4] = (vic_pv[1:, :, :2] - vic_pv[:-1, :, :2]) / dt
    vic_pv[-1, :, 2:4] = vic_pv[-2, :, 2:4]
    return vic_pv



