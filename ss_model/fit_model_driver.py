import os
from utils.constants import DATASETS_ROOT, SAMPLE_DATASETS_ROOT
import utils.tt_dataset as tt
import display.data_explore as de
from ss_model import data_selection as ds
from ss_model import estimate_kalman as ek
from ss_model import estimate_model as em


def main_driver():
    # data_dir = os.path.join(DATASETS_ROOT, 'tt_format/10hz/dut')
    data_dir = os.path.join(SAMPLE_DATASETS_ROOT, 'tt_format/10hz/dut')
    # n_obs, n_pred do not matter for fitting
    n_obs = 50
    n_pred = 50
    dataset = tt.process_data2datasets(
        data_dir, n_obs, n_pred,
        dataset_id2file_id_list={0: range(0, 35)},
        is_fill_nan=True,
        valid_ids_kwargs=dict(exist_any_type=(tt.AgentType.ped,)),
    )
    print('Loaded')

    DT = 0.1  # time (s) of sampling interval
    SIGMA_X = 0.05  # sd (m) of observed positions
    # (lx, ly, dx, dy, nx, ny)
    # 2D grid of l + range(n)*d
    RT_GRID = (0, 0, 0.4, 0.4, 5, 5)
    # (l, m): 1D grid of l +
    MRX_GRID = (6., 7)  # (l, m) : 1D grid [0, ..., l].size = m
    ds.dataset_filter_minimum_obs(dataset, tau_frames=3)
    ds.dataset_filter_stationary_vic(dataset, DT, tau_s=0.5)
    ds.dataset_set_r_and_is_obs_frames(
        dataset, DT, MRX_GRID[0], tau_frames=3, is_rm_unknown=True)
    print('Heuristics calculated')

    A, Q, C, R, sigma_0 = ek.make_ss_matrices(SIGMA_X, DT)
    D = ek.make_state2v_matrix()
    # estimate
    # - sd (m/s) of internal velocity drift
    # - smoothed estimates of state for 'observed' timesteps
    # - smoothed estimates of velocity for 'unobserved' timesteps
    sigma_v_hat = ek.estimate_dataset_x_sigmaq_v_runs_coord_descent(
        A, Q, C, R, D, dataset, sigma_0, n_iter=5)
    print('- sigma_v -')
    print(sigma_v_hat)

    em.estimate_x_unobs(dataset)
    em.set_vic_sm_cv(dataset, DT)
    mrx, Z, v_hat, v, df_id2inds = em.make_unobs_design_matrices(
        dataset, DT, MRX_GRID, RT_GRID)
    print('Design matrices built')
    # print(mrx.round(2))
    print('--')

    u, beta, q = em.estimate_u_beta_q_em(mrx, Z, v_hat, v, SIGMA_X, DT, n_iter=20)

    import numpy as np
    np.set_printoptions(suppress=True)
    print('u [0 -> l]:')
    print(u.round(decimals=2))
    print('beta as grid [(row,col)=(r, tau)=(lx, ly) top left]:')
    print(beta[1:].reshape(RT_GRID[4], RT_GRID[5]).round(decimals=2))
    print('beta_0 = {:.2f}'.format(beta[0]))
    print('fraction of q=1: {:.2f}'.format(q.mean()))
    exit(0)

    em.set_q_estimates(dataset, df_id2inds, q)
    # de.display_dataset_qr_per_agent(dataset, df_id2inds)
    de.display_dataset_qr_per_frame_window(
        dataset, df_id2inds, 100, n_skip=50,
        is_draw_r=True)





if __name__ == '__main__':
    main_driver()
