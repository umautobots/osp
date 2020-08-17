import numpy as np
from utils.tt_dataset import AgentType, TrajectoryTypeDataset
from test_problems.u_grid_encoding import xy2mrx_v1 as xy2mrx
from test_problems.risk_distances import dist2rt_v1 as dist2rt
from test_problems.grid_encoding import rt2enc_v1 as rt2enc
from utils import general as ge
import scipy.optimize as opt


def estimate_x_unobs(dataset):
    # use \hat{x} as unbiased estimate
    for df_info in dataset.df_list:
        df = df_info.df
        mask = ~df.is_obs & df.type_id == AgentType.ped
        df.loc[mask, ['sm_x', 'sm_y']] = df.loc[mask, ['x', 'y']].values


def set_vic_sm_cv(dataset, dt):
    """
    Set sm_[x y vx vy] for vic
    Velocity via constant velocity (from last)
    assume observations are continuous
    :param dataset: exist [frame_id(index) agent_id type_id x y
        sm_x sm_y sm_vx sm_vy]
    :param dt:
    :return:
    """
    for df_info in dataset.df_list:
        df = df_info.df
        vic_ids = df.loc[df.type_id == AgentType.vic, 'agent_id'].unique()
        for vic_id in vic_ids:
            vic_df = df.loc[df.agent_id == vic_id]
            v = (vic_df[['x', 'y']].values[1:] -
                 vic_df[['x', 'y']].values[:-1]) / dt
            v = np.vstack((v, v[-1, :]))
            df.loc[df.agent_id == vic_id, ['sm_vx', 'sm_vy']] = v
            df.loc[df.agent_id == vic_id, ['sm_x', 'sm_y']] =\
                vic_df[['x', 'y']].values


class DataframeInds:
    """
    For later reassigning vector's values as
    df.loc[frame_id in frames &
           agent_id == agent_id, 'q'] = q[q_inds]
    """

    def __init__(self, n_q_offset=0):
        self.frames_list = []
        self.agent_ids = []
        self.q_inds_list = []
        self._n_q = n_q_offset

    def append(self, frames, agent_id, q_inds, is_relative=False):
        self.frames_list.append(frames)
        self.agent_ids.append(agent_id)
        add_inds = q_inds
        if is_relative:
            add_inds = add_inds + self._n_q
        self.q_inds_list.append(add_inds)
        self._n_q += len(q_inds)

    def get_offset(self):
        return self._n_q

    def __len__(self):
        return len(self.frames_list)


def make_unobs_design_matrices(dataset, dt, mrx_grid, rt_grid):
    """
    for data term:
    \lVert [q \kron (1 \\ 1)] \circ v +
           [(1-q) \kron (1 \\ 1)] \circ v \circ (Au) -
           \hat{v} \rVert_2^2
    so A = (...\\ m_{r_t}(x_t) \\ ...) \kron (1 \\ 1)
    ie the matrix that picks out which u are 'active' during this 'slowing' timestep
    - order by [dataset.df_list, agent_id, frame_id]
    - assume ~is_obs => r >= 0
    :param dataset: [sm_(x y vx vy] set for all agents, r]
    :param mrx_grid:
    :param rt_grid:
    :return:
    """
    cols = ('sm_x', 'sm_y', 'sm_vx', 'sm_vy')
    mrx_list = []
    z_list = []
    v_hat = []
    v = []
    df_id2inds = {}
    n_q_offset = 0
    for df_info in dataset.df_list:
        df = df_info.df
        df_inds = DataframeInds(n_q_offset=n_q_offset)
        print('df unobs total 2: ', (~df['is_obs'].values).sum())
        ped_ids = np.unique(df.loc[df.type_id == AgentType.ped, 'agent_id'].values)
        for ped_id in ped_ids:
            # set \hat{v} = x_next - x_prev / dt
            ped_df = df.loc[df.agent_id == ped_id].copy()
            ped_frames = ped_df.index.unique().values
            ped_df['vh_x'] = np.nan
            ped_df['vh_y'] = np.nan
            ped_df.loc[ped_df.index[:-1], 'vh_x'] =\
                (ped_df['x'].values[1:] - ped_df['x'].values[:-1]) / dt
            ped_df.loc[ped_df.index[:-1], 'vh_y'] =\
                (ped_df['y'].values[1:] - ped_df['y'].values[:-1]) / dt
            unobs_ped_df = ped_df.loc[~ped_df.is_obs].iloc[:-1]
            frames = unobs_ped_df.index.unique().values
            if frames.size == 0:
                # print('skipped')
                continue
            frame_ptns = ge.split_to_consecutive_v0(frames)
            # single frames can have seq_r = [-1]
            frame_ptns = [frame_ptn for frame_ptn in frame_ptns if len(frame_ptn) > 1]
            if len(frame_ptns) == 0:
                continue

            seq_df_all = df.loc[ped_frames]
            seq_df_all = seq_df_all.loc[(seq_df_all.agent_id == ped_id) |
                                        (seq_df_all.type_id == AgentType.vic)]
            frames_all = seq_df_all.index.unique().values
            ped_pv_all, vic_pv_all = TrajectoryTypeDataset.build_nan_df(
                seq_df_all, frames_all[0], frames_all.size, cols=cols)
            for frame_ptn in frame_ptns:
                # print(frame_ptn)
                # seq_df = seq_df_all.loc[frame_ptn]
                # print(seq_df.loc[seq_df.agent_id == ped_id, 'is_r_all'].values)
                # print(seq_df.loc[seq_df.agent_id == ped_id, 'r'].values)
                # print(seq_df.loc[seq_df.agent_id == ped_id, 'is_obs'].values)
                # print(seq_df.loc[seq_df.agent_id == ped_id])

                frames_all_inds = np.arange(frame_ptn.size) + int(frame_ptn[0] - frames_all[0])
                seq_r = seq_df_all.loc[seq_df_all.agent_id == ped_id, 'r']\
                    .values.astype(np.int)[frames_all_inds]
                assert np.all(seq_r >= 0), seq_r
                # select via r -> n_frames, 4
                vic_pv = vic_pv_all[frames_all_inds, seq_r, :]
                ped_pv = ped_pv_all[frames_all_inds, 0, :]
                frames_range = np.arange(frame_ptn.size)
                mrx_rows = xy2mrx(ped_pv[:, :2], vic_pv, mrx_grid)

                rt = dist2rt(ped_pv, vic_pv)
                rt = np.log10(rt)
                rt[:, :, 0] /= 2
                z_rows = rt2enc(rt, rt_grid)  # n_ped=n_frames, n_vic=n_frames, n_rt
                z_rows = z_rows[frames_range, frames_range, :]

                mrx_list.append(mrx_rows)
                z_list.append(z_rows)
                v_hat.append(unobs_ped_df.loc[frame_ptn, ['vh_x', 'vh_y']].values)
                v.append(unobs_ped_df.loc[frame_ptn, ['sm_vx', 'sm_vy']].values)
                df_inds.append(frame_ptn, ped_id, frames_range, is_relative=True)
        if len(df_inds) > 0:
            df_id2inds[df_info.datafile_path] = df_inds
            n_q_offset = df_inds.get_offset()
    mrx = np.vstack(mrx_list)
    z = np.vstack(z_list)
    z = np.hstack((1+0*z[:, [0]], z))  # constant
    v_hat = np.vstack(v_hat).T.ravel(order='F')
    v = np.vstack(v).T.ravel(order='F')
    print(mrx.shape)
    print(z.shape)
    print(v_hat.shape)
    return mrx, z, v_hat, v, df_id2inds


def estimate_u_beta_q_em(mrx, Z, v_hat, v, sigma_x, dt, n_iter=10):
    q = initialize_q_em_v0(v_hat, v)
    beta = ()
    u = ()
    precision_u = 1/20
    precision_beta = 1/10

    def f_obj(q_err_, u_, beta_):
        mean_err = q_err_ + ((precision_u * u_) ** 2).sum() + \
                   ((precision_beta * beta_) ** 2).sum()
        return mean_err / q.size

    for _ in range(n_iter):
        u = u_given_q(q, mrx, v_hat, v, sigma_x, dt, precision_u)
        beta = beta_given_q(q, Z, beta=beta, precision_beta=precision_beta)
        q, q_err = q_given_u_beta(u, beta, mrx, v_hat, v, Z, sigma_x, dt)
        print('f = {:.4f}'.format(f_obj(q_err, u, beta)))
    return u, beta, q


def estimate_u_beta_q_em_v1(mrx, Z, v_hat, v, sigma_x, dt, n_iter=10):
    q = initialize_q_em_v0(v_hat, v)
    beta = ()
    u = ()
    precision_u = 1.  #/20
    precision_beta = 1/10

    def f_obj(q_err_, u_, beta_):
        n_u = u.size
        C = np.eye(n_u - 1) + np.diag(-np.ones(n_u - 2), k=-1)
        bc = np.zeros(n_u - 1)
        bc[0] = -u_[0]
        mean_err = q_err_ + ((precision_beta * beta_) ** 2).sum() + \
            (precision_u * np.linalg.norm(C.dot(u_[1:]) - bc))**2
        return mean_err / q.size

    for _ in range(n_iter):
        u = u_given_q_v1(q, mrx, v_hat, v, sigma_x, dt, precision_u)
        beta = beta_given_q(q, Z, beta=beta, precision_beta=precision_beta)
        q, q_err = q_given_u_beta(u, beta, mrx, v_hat, v, Z, sigma_x, dt)
        print('f = {:.4f}'.format(f_obj(q_err, u, beta)))
    return u, beta, q


def initialize_q_em_v0(v_hat, v):
    v_norm = np.linalg.norm(v.reshape(-1, 2), axis=1)
    v_hat_norm = np.linalg.norm(v_hat.reshape(-1, 2), axis=1)
    q = (v_hat_norm >= 0.2 * v_norm) * 1.
    # q = (np.random.randn(q.size) > 0) * 1.  # also works
    return q


def u_given_q(q, mrx, v_hat, v, sigma_x, dt, precision_u):
    """
    Estimate field parameters u
    - st. -1 <= u <= 1

    \lVert (v \circ \{([1-q] \circ mrx) \kron (1 \\ 1)\}) u -
            \hat{v} \circ ([1-q] \kron (1 \\ 1)) \rVert_2^2 * (dt^2)/2*sigma_x^2

    :param q: n, |
    :param mrx: n, n_u | design matrix for u grid
    :param v_hat: 2n,  | differenced velocities (may be slowing)
    :param v: 2n, | actual desired velocities (estimated)
    :param sigma_x:
    :param dt:
    :param precision_u:
    :return:
        u: n_u, |
    """
    n_u = mrx.shape[1]
    A = (mrx.T * (1-q)).T
    A = np.repeat(A, 2, axis=0)
    A = (A.T * v).T * dt / sigma_x
    b = np.repeat((1-q), 2) * v_hat * dt / sigma_x
    # res = np.linalg.lstsq(A, b, rcond=None)[0]
    # add shrinkage + in case close ranges of u not seen
    As = np.vstack((A, np.eye(n_u) * precision_u))
    bs = np.hstack((b, np.zeros(n_u)))
    u = np.linalg.lstsq(As, bs, rcond=None)[0]
    if 1 < u.max():
        print('1 < u_max, ', u.max())
    if u.min() < -1:
        print('u_min < -1, ', u.min())
    np.clip(u, -1, 1, out=u)
    return u


def u_given_q_v1(q, mrx, v_hat, v, sigma_x, dt, precision_u):
    """
    Estimate field parameters u, setting close one to -1
    - st. -1 <= u <= 1
    - u_0 = -1
    - ||Cu||_2^2 <=> each u_i close to u_{i+1}
      ie add term (u has all but u_0)
      Cu - b_c = (-u_1 \\ u_1-u_2 \\ ...) - (-u_0 \\ 0 \\ ...)
      with u_0 = -1
    - and A' = A[:, 1:], b' = b - u_0*A[:, 0]

    \lVert (v \circ \{([1-q] \circ mrx) \kron (1 \\ 1)\}) u -
            \hat{v} \circ ([1-q] \kron (1 \\ 1)) \rVert_2^2 * (dt^2)/2*sigma_x^2

    :param q: n, |
    :param mrx: n, n_u | design matrix for u grid
    :param v_hat: 2n,  | differenced velocities (may be slowing)
    :param v: 2n, | actual desired velocities (estimated)
    :param sigma_x:
    :param dt:
    :param precision_u:
    :return:
        u: n_u, |
    """
    n_u = mrx.shape[1]
    A = (mrx.T * (1-q)).T
    A = np.repeat(A, 2, axis=0)
    A = (A.T * v).T * dt / sigma_x
    b = np.repeat((1-q), 2) * v_hat * dt / sigma_x
    u_0 = -1
    C = np.eye(n_u-1) + np.diag(-np.ones(n_u-2), k=-1)
    bc = np.zeros(n_u-1)
    bc[0] = -u_0
    As = np.vstack((A[:, 1:], C * precision_u))
    bs = np.hstack((b - u_0 * A[:, 0], bc * precision_u))
    u = np.linalg.lstsq(As, bs, rcond=None)[0]
    u = np.hstack((u_0, u))
    if 1 < u.max():
        print('1 < u_max, ', u.max())
    if u.min() < -1:
        print('u_min < -1, ', u.min())
    np.clip(u, -1, 1, out=u)
    return u


def beta_given_q(q, Z, beta=(), precision_beta=0.):
    # add very small shrinkage
    beta = beta if len(beta) > 0 else np.zeros((Z.shape[1],), dtype=np.float)
    res = opt.minimize(
        lambda x: logistic_obj(x, q, Z).sum() + (x**2).sum() * precision_beta**2,
        beta, method='BFGS')
    beta = res.x
    return beta


def logistic_obj(beta, q, Z):
    """
    Calculate f_i for
    f(beta) = 1/n sum_{i=1}^n f_i
    with
    f_i = -[q_i log(h_i) + (1-q_i) log(1-h_i)]
    h_i = 1/(1 + exp{-beta'z_i})
    :param beta: m,
    :param q: n,
    :param Z: n, m
    :return:
    """
    y = -Z.dot(beta)
    # log_h = -np.log(1 + np.exp(y))
    log_h = -np.logaddexp(0*y, y)
    log_1mh = y + log_h
    f = -(q * log_h + (1 - q) * log_1mh)
    return f


def q_given_u_beta(u, beta, mrx, v_hat, v, Z, sigma_x, dt):
    q_ones = np.ones((Z.shape[0],), dtype=np.float)
    logistic_0 = logistic_obj(beta, 0*q_ones, Z)
    logistic_1 = logistic_obj(beta, q_ones, Z)
    A = np.repeat(mrx, 2, axis=0)
    scaling = 0.5 * (dt / sigma_x) ** 2
    l2_err_0 = ((v*(A.dot(u)) - v_hat).reshape(-1, 2) ** 2).sum(axis=1) * scaling
    l2_err_1 = ((v - v_hat).reshape(-1, 2) ** 2).sum(axis=1) * scaling
    q = (l2_err_1 + logistic_1 <= l2_err_0 + logistic_0) * 1.
    q_err = (l2_err_1 + logistic_1) * q + (l2_err_0 + logistic_0) * (1-q)
    return q, q_err.sum()


def set_q_estimates(dataset, df_id2inds, q):
    for df_info in dataset.df_list:
        df = df_info.df
        df['q'] = np.nan
        if df_info.datafile_path not in df_id2inds:
            continue
        df_inds = df_id2inds[df_info.datafile_path]
        for i in range(len(df_inds)):
            mask = (df.index.isin(df_inds.frames_list[i])) &\
                   (df.agent_id == df_inds.agent_ids[i])
            df.loc[mask, 'q'] = q[df_inds.q_inds_list[i]]

