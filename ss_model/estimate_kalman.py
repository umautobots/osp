import numpy as np
from scipy import sparse
from scipy.sparse import linalg as la
from scipy import optimize as opt
from ss_model import linear_operators as lo
from utils.tt_dataset import AgentType


def make_ss_matrices(sigma_x, dt):
    """
    To make Q full-rank for inversion (so the mle makes sense), use:
    Q = [ dt**2  dt/2
            dt/2    1 ]
    to approximate Q = (dt 1)(dt 1)'
    System:
    x = [p_x p_y v_x v_y]
    y = [p_x' p_y']
    :param sigma_x: 
    :param dt: 
    :return: 
        sigma_0: starting value for sigma_v, with process variance (sigma_v^2 Q)
    """
    i2 = np.eye(2)
    _ = np.zeros((2, 2))
    A = np.block([
        [i2, dt*i2],
        [_, i2],
    ])
    Q = np.block([
        [dt**2 * i2, dt*i2 * .5],
        [dt*i2 * .5, i2],
    ])
    C = np.block([i2, _])
    R = sigma_x**2 * i2
    sigma_0 = float(sigma_x) / 2
    return A, Q, C, R, sigma_0


def make_state2v_matrix():
    i2 = np.eye(2)
    _ = np.zeros((2, 2))
    D = np.block([_, i2])
    return D


class DatasetInds:

    def __init__(self, n_total_vars_all=0):
        self.x = []
        self.v = []
        self.x_next = []
        self.v_next = []
        self.x_enter = []
        self.v_enter = []
        self.x_exit = []
        self.v_exit = []
        self.n_total_vars = n_total_vars_all  # serve as offset
        self.inds_list = [
            self.x, self.v, self.x_next, self.v_next,
            self.x_enter, self.v_enter, self.x_exit, self.v_exit]

    def append(self, x, v, x_next, v_next, x_enter, v_enter,
               x_exit, v_exit, n_total_vars):
        # inds to add are wrt 0, so add current n_total_vars offset
        add_list = [x, v, x_next, v_next, x_enter, v_enter, x_exit, v_exit]
        for l, add_l in zip(self.inds_list, add_list):
            if len(add_l) > 0:
                l.append(add_l + self.n_total_vars)
        self.n_total_vars += n_total_vars

    def hstack(self):
        self.x, self.v, self.x_next, self.v_next,\
            self.x_enter, self.v_enter, self.x_exit, self.v_exit = [
                np.hstack(l) if len(l) > 0 else np.array([])
                for l in self.inds_list]


def estimate_dataset_x_sigmaq_v_runs_coord_descent(
        A, Q, C, R, D, dataset, sigma_0, n_iter=10):
    """
    Estimates x where df.is_obs and all v (including runs between observed x)
    Sets (smoothing) estimates to columns ['sm_x', 'sm_y', 'sm_vx', 'sm_vy']
    :param A: n, n
    :param Q: n, n
    :param C: m, n
    :param R: m, m
    :param D: n_v, n | selects out v from x (mainly, x' = (x y vx vy))
    :param dataset: exist [frame_id(index), agent_id, x, y, is_obs]
    :param sigma_0:
    :param n_iter:
    :return:
    """
    n_v, n = D.shape
    Qi = np.linalg.inv(Q)
    Qih = np.linalg.cholesky(Qi).T
    Ri = np.linalg.inv(R)
    Rih = np.linalg.cholesky(Ri).T
    df_id2inds = {}
    df_id2y = {}
    sigma_part, no_sigma_part, b = [], [], []
    n_total_vars_all = 0
    for df_info in dataset.df_list:
        df = df_info.df
        df_inds = DatasetInds(n_total_vars_all)
        y_ordered = []
        ped_ids = np.unique(df.loc[df.type_id == AgentType.ped, 'agent_id'].values)
        for ped_id in ped_ids:
            ped_df = df.loc[df.agent_id == ped_id]
            reps = n * ped_df['is_obs'].values + n_v * (1 - ped_df['is_obs'].values)
            n_total_vars = reps.sum()
            x_mask = np.repeat(ped_df['is_obs'].values, reps)
            inds_x_i = np.arange(n_total_vars)[x_mask]
            inds_v_i = np.arange(n_total_vars)[~x_mask]
            sigma_part_i = []

            inds_x_next = inds_x_i[n::n][(inds_x_i[n::n] - inds_x_i[:-n:n]) == n]
            n_x_next = inds_x_next.size
            inds_x_next = np.tile(np.arange(n), n_x_next) + np.repeat(inds_x_next, n)
            inds_x_prev = inds_x_next - n
            sigma_part_i.append(
                lo.left_mult_mat(Qih, n_x_next) *
                lo.select_inds(n_total_vars, inds_x_next) -
                lo.left_mult_mat(Qih.dot(A), n_x_next) *
                lo.select_inds(n_total_vars, inds_x_prev)
            )
            b.append(np.zeros(n * n_x_next))
            inds_v_next = inds_v_i[n_v::n_v][(inds_v_i[n_v::n_v] - inds_v_i[:-n_v:n_v]) == n_v]
            n_v_next = inds_v_next.size
            if n_v_next > 0:
                inds_v_next = np.tile(np.arange(n_v), n_v_next) + np.repeat(inds_v_next, n_v)
                inds_v_prev = inds_v_next - n_v
                sigma_part_i.append(
                    lo.select_inds(n_total_vars, inds_v_next) -
                    lo.select_inds(n_total_vars, inds_v_prev)
                )
                b.append(np.zeros(n_v * n_v_next))
            inds_x_enter = inds_x_i[:-n:n][(inds_x_i[n::n] - inds_x_i[:-n:n]) != n]
            if n_v_next > 0 and inds_x_i[-1] < inds_v_i[-1]:
                inds_x_enter = np.hstack((inds_x_enter, inds_x_i[-n]))
            n_x_enter = inds_x_enter.size
            inds_v_enter = []
            if n_x_enter > 0:
                inds_v_enter = np.tile(np.arange(n_v), n_x_enter) +\
                               np.repeat(inds_x_enter + n, n_v)
                inds_x_enter = np.tile(np.arange(n), n_x_enter) + np.repeat(inds_x_enter, n)
                sigma_part_i.append(
                    lo.select_inds(n_total_vars, inds_v_enter) -
                    lo.left_mult_mat(D, n_x_enter) *
                    lo.select_inds(n_total_vars, inds_x_enter)
                )
                b.append(np.zeros(n_v * n_x_enter))
            inds_x_exit = inds_x_i[n::n][(inds_x_i[n::n] - inds_x_i[:-n:n]) != n]
            if n_v_next > 0 and inds_v_i[0] < inds_x_i[0]:
                inds_x_exit = np.hstack((inds_x_i[0], inds_x_exit))
            n_x_exit = inds_x_exit.size
            inds_v_exit = []
            if n_x_exit > 0:
                inds_v_exit = np.tile(np.arange(n_v), n_x_exit) +\
                              np.repeat(inds_x_exit - n_v, n_v)
                inds_x_exit = np.tile(np.arange(n), n_x_exit) + np.repeat(inds_x_exit, n)
                sigma_part_i.append(
                    lo.left_mult_mat(D, n_x_exit) *
                    lo.select_inds(n_total_vars, inds_x_exit) -
                    lo.select_inds(n_total_vars, inds_v_exit)
                )
                b.append(np.zeros(n_v * n_x_exit))
            sigma_part.append(lo.vstack(sigma_part_i))
            no_sigma_part.append(
                lo.left_mult_mat(Rih.dot(C), int(inds_x_i.size/n)) *
                lo.select_inds(n_total_vars, inds_x_i)
            )
            b.append(Rih.dot(
                ped_df.loc[ped_df.is_obs, ['x', 'y']].T).ravel(order='F'))
            y_ordered.append(ped_df.loc[ped_df.is_obs, ['x', 'y']].T)
            df_inds.append(
                inds_x_i, inds_v_i, inds_x_next, inds_v_next,
                inds_x_enter, inds_v_enter, inds_x_exit, inds_v_exit, n_total_vars
            )
        df_inds.hstack()
        df_id2inds[df_info.datafile_path] = df_inds
        df_id2y[df_info.datafile_path] = np.hstack(y_ordered)
        n_total_vars_all = df_inds.n_total_vars
    b = np.hstack(b)

    def x_given_sigma(sigma_q, x0=()):
        x0 = x0 if len(x0) > 0 else None
        Ap = lo.block_diag([lo.vstack(
            [sigma_part[i_] * (1/sigma_q), no_sigma_part[i_]])
            for i_ in range(len(sigma_part))
        ])
        res = la.lsqr(Ap, b, x0=x0)
        print('lsqr norm ', res[3])
        return res[0]

    def sigma_given_x(x):
        weighted_err = 0.
        bot = 0.
        for df_id, df_inds in df_id2inds.items():
            x_err = x[df_inds.x_next].reshape(n, -1, order='F') -\
                    A.dot(x[df_inds.x_next - n].reshape(n, -1, order='F'))
            weighted_err += (x_err * Qi.dot(x_err)).sum()
            n_v_err = df_inds.v_next.size/n_v
            if n_v_err > 0:
                v_err = x[df_inds.v_next] - x[df_inds.v_next - n_v]
                weighted_err += (v_err ** 2).sum()
            n_x_enter_err = df_inds.v_enter.size/n_v
            if n_x_enter_err > 0:
                x_enter_err = x[df_inds.v_enter].reshape(n_v, -1, order='F') -\
                              D.dot(x[df_inds.x_enter].reshape(n, -1, order='F'))
                weighted_err += (x_enter_err ** 2).sum()
            n_x_exit_err = df_inds.v_exit.size/n_v
            if n_x_exit_err > 0:
                x_exit_err = D.dot(x[df_inds.x_exit].reshape(n, -1, order='F')) -\
                             x[df_inds.v_exit].reshape(n_v, -1, order='F')
                weighted_err += (x_exit_err ** 2).sum()
            bot += n * x_err.shape[1] + n_v * (n_v_err + n_x_enter_err + n_x_exit_err)
        sigma_sq = weighted_err / bot
        return np.sqrt(sigma_sq)

    def f_fcn(x, sigma):
        weighted_err = 0.
        y_loss = 0.
        bot = 0.
        for df_id, df_inds in df_id2inds.items():
            x_err = x[df_inds.x_next].reshape(n, -1, order='F') -\
                    A.dot(x[df_inds.x_next - n].reshape(n, -1, order='F'))
            weighted_err += (x_err * Qi.dot(x_err)).sum()
            n_v_err = df_inds.v_next.size/n_v
            if n_v_err > 0:
                v_err = x[df_inds.v_next] - x[df_inds.v_next - n_v]
                weighted_err += (v_err ** 2).sum()
            n_x_enter_err = df_inds.v_enter.size/n_v
            if n_x_enter_err > 0:
                x_enter_err = x[df_inds.v_enter].reshape(n_v, -1, order='F') -\
                              D.dot(x[df_inds.x_enter].reshape(n, -1, order='F'))
                weighted_err += (x_enter_err ** 2).sum()
            n_x_exit_err = df_inds.v_exit.size/n_v
            if n_x_exit_err > 0:
                x_exit_err = D.dot(x[df_inds.x_exit].reshape(n, -1, order='F')) -\
                             x[df_inds.v_exit].reshape(n_v, -1, order='F')
                weighted_err += (x_exit_err ** 2).sum()
            bot += n * x_err.shape[1] + n_v * (n_v_err + n_x_enter_err + n_x_exit_err)
            y_err = df_id2y[df_id] - C.dot(x[df_inds.x].reshape(n, -1, order='F'))
            y_loss += 0.5 * (y_err * Ri.dot(y_err)).sum()
        print('f fcn ', 0.5 * weighted_err / (sigma ** 2), y_loss, bot * np.log(sigma))
        fx = 0.5 * weighted_err / (sigma ** 2) + y_loss + bot * np.log(sigma)
        return fx

    sigma_q = sigma_0
    x = ()
    for i in range(n_iter):
        x = x_given_sigma(sigma_q, x0=x)
        print(f_fcn(x, sigma_q))
        sigma_q = sigma_given_x(x)
        print(f_fcn(x, sigma_q), sigma_q)
    set_column_estimates(dataset, df_id2inds, x)
    return sigma_q


def set_column_estimates(dataset, df_id2inds, x):
    n, n_v = 4, 2
    for df_info in dataset.df_list:
        df = df_info.df
        df.reset_index(inplace=True)
        df.sort_values(by=['agent_id', 'frame_id'], inplace=True)
        df_inds = df_id2inds[df_info.datafile_path]
        for col in ['sm_x', 'sm_y', 'sm_vx', 'sm_vy']:
            df[col] = np.nan
        is_ped = df.type_id == AgentType.ped
        df.loc[df.is_obs & is_ped, ['sm_x', 'sm_y']] =\
            x[df_inds.x].reshape(n, -1, order='F')[:2, :].T
        df.loc[df.is_obs & is_ped, ['sm_vx', 'sm_vy']] =\
            x[df_inds.x].reshape(n, -1, order='F')[2:, :].T
        df.loc[~df.is_obs & is_ped, ['sm_vx', 'sm_vy']] =\
            x[df_inds.v].reshape(n_v, -1, order='F').T
        df.set_index('frame_id', inplace=True)
        df.sort_index(inplace=True)


def estimate_blocks_lo_x_sigmaq_coord_descent(A, Q, C, R, y_blocks, sigma_0, n_iter=10):
    """
    Using linear operators for estimate_blocks_x_sigmaq_coord_descent
    :param A: n, n
    :param Q: n, n
    :param C: m, n
    :param R: m, m
    :param y_blocks: list[i] = m, T_i | blocks of sequential noisy observations, each T_i >= 2
    :param sigma_0: initial value of sigma
    :param n_iter:
    :return:
        sigma_q:
        x: n, st = sum_{i=1,..N} T_i
    """
    Qi = np.linalg.inv(Q)
    Qih = np.linalg.cholesky(Qi).T
    Ri = np.linalg.inv(R)
    Rih = np.linalg.cholesky(Ri).T
    t_csum = np.cumsum([0] + [yi.shape[1] for yi in y_blocks])
    st = t_csum[-1]
    n_blocks = len(y_blocks)
    y = np.hstack(y_blocks)
    n = A.shape[0]
    A_x, A_y, b = [], [], []
    for yi in y_blocks:
        ti = yi.shape[1]
        A_x.append(
            lo.left_mult_mat(Qih, ti - 1) *
            lo.select_inds(n*ti, np.arange(n, n*ti)) -
            lo.left_mult_mat(Qih.dot(A), ti - 1) *
            lo.select_inds(n*ti, np.arange(n*(ti-1)))
        )
        A_y.append(lo.left_mult_mat(Rih.dot(C), ti))
        b.append(np.hstack((
            np.zeros((n * (ti - 1))),
            Rih.dot(yi).T.ravel()
        )))
    b = np.hstack(b)

    def x_given_sigma(sigma_q, x0=()):
        x0 = x0.ravel(order='F') if len(x0) > 0 else None
        Ap = lo.block_diag([lo.vstack(
            [A_x[i_] * (1/sigma_q), A_y[i_]]) for i_ in range(n_blocks)
        ])
        res = la.lsqr(Ap, b, x0=x0)
        x = np.ascontiguousarray(res[0].reshape((n, st), order='F'))
        return x

    def sigma_given_x(x):
        weighted_err = 0.
        for j in range(n_blocks):
            x_err = x[:, t_csum[j]+1:t_csum[j+1]] -\
                    A.dot(x[:, t_csum[j]:t_csum[j+1]-1])
            weighted_err += (x_err * Qi.dot(x_err)).sum()
        sigma_sq = weighted_err/(n * (st - n_blocks))
        return np.sqrt(sigma_sq)

    f_fcn = lambda x, sigma: x_sigmaq_blocks_f_fcn(x, sigma, n_blocks, t_csum, A, Qi, C, Ri, y)
    sigma_q = sigma_0
    x = ()
    for i in range(n_iter):
        x = x_given_sigma(sigma_q, x0=x)
        print(f_fcn(x, sigma_q))
        sigma_q = sigma_given_x(x)
        print(f_fcn(x, sigma_q), sigma_q)
    return sigma_q, x


def estimate_blocks_x_sigmaq_coord_descent(A, Q, C, R, y_blocks, sigma_0, n_iter=10):
    """
    Version of estimate_x_sigmaq_coord_descent for blocks of observations with gaps in between
    :param A: n, n
    :param Q: n, n
    :param C: m, n
    :param R: m, m
    :param y_blocks: list[i] = m, T_i | blocks of sequential noisy observations, each T_i >= 2
    :param sigma_0: initial value of sigma
    :param n_iter: 
    :return: 
        sigma_q:
        x: n, st = sum_{i=1,..N} T_i
    """
    Qi = np.linalg.inv(Q)
    Qih = np.linalg.cholesky(Qi).T
    Ri = np.linalg.inv(R)
    Rih = np.linalg.cholesky(Ri).T
    t_csum = np.cumsum([0] + [yi.shape[1] for yi in y_blocks])
    st = t_csum[-1]
    n_blocks = len(y_blocks)
    y = np.hstack(y_blocks)
    n = A.shape[0]
    L_a, R_a, L_y, b = [], [], [], []
    for yi in y_blocks:
        ti = yi.shape[1]
        L_a.append(sparse.kron(
            sparse.eye(ti, ti, format='lil')[1:, :],
            sparse.lil_matrix(Qih),
            format='csr'
        ))
        R_a.append(sparse.kron(
            sparse.eye(ti, ti, format='lil')[:-1, :],
            sparse.lil_matrix(Qih.dot(A)),
            format='csr'
        ))
        L_y.append(sparse.kron(
            sparse.eye(ti, ti, format='csr'),
            sparse.csr_matrix(Rih.dot(C)),
        ))
        b.append(np.hstack((
            np.zeros((n * (ti - 1))),
            Rih.dot(yi).T.ravel()
        )))
    b = np.hstack(b)

    def x_given_sigma(sigma_q, x0=()):
        x0 = x0.ravel(order='F') if len(x0) > 0 else None
        Ap = sparse.block_diag(
            [sparse.vstack(((L_a[i_] - R_a[i_]) / sigma_q, L_y[i_]))
             for i_ in range(n_blocks)],
            format='csr'
        )
        res = la.lsqr(Ap, b, x0=x0)
        x = np.ascontiguousarray(res[0].reshape((n, st), order='F'))
        return x

    def sigma_given_x(x):
        weighted_err = 0.
        for j in range(n_blocks):
            x_err = x[:, t_csum[j]+1:t_csum[j+1]] -\
                    A.dot(x[:, t_csum[j]:t_csum[j+1]-1])
            weighted_err += (x_err * Qi.dot(x_err)).sum()
        sigma_sq = weighted_err/(n * (st - n_blocks))
        return np.sqrt(sigma_sq)

    f_fcn = lambda x, sigma: x_sigmaq_blocks_f_fcn(x, sigma, n_blocks, t_csum, A, Qi, C, Ri, y)
    sigma_q = sigma_0
    x = ()
    for i in range(n_iter):
        x = x_given_sigma(sigma_q, x0=x)
        print(f_fcn(x, sigma_q))
        sigma_q = sigma_given_x(x)
        print(f_fcn(x, sigma_q), sigma_q)
    return sigma_q, x


def x_sigmaq_blocks_f_fcn(x, sigma, n_blocks, t_csum, A, Qi, C, Ri, y):
    """

    :param x: n, sum_{i=1,..N} T_i
    :param sigma: of Q_{full} = sigma^2 Q
    :param n_blocks:
    :param t_csum: [0, T_1, T_1 + T_2, ... ]
    :param A:
    :param Qi: Q^-1
    :param C:
    :param Ri: R^-1
    :param y: m, sum_{i=1,..N} T_i
    :return:
    """
    n = A.shape[0]
    st = t_csum[-1]
    weighted_err = 0.
    for j in range(n_blocks):
        x_err = x[:, t_csum[j] + 1:t_csum[j + 1]] - \
                A.dot(x[:, t_csum[j]:t_csum[j + 1] - 1])
        weighted_err += (x_err * Qi.dot(x_err)).sum()
    a_loss = 0.5 * weighted_err / (sigma ** 2)
    y_err = y - C.dot(x)
    y_loss = 0.5 * (y_err * Ri.dot(y_err)).sum()
    fx = a_loss + y_loss + n * (st - n_blocks) / 2 * 2 * np.log(sigma)
    return fx


def estimate_x_sigmaq_coord_descent(A, Q, C, R, y, sigma_0, x0=(), n_iter=10):
    """
    System equations:
        x_t = Ax_{t-1} + sigma_v (Q^.5) w_t
        y_t = Cx_t + (R^.5) v_t
    with
        w_t, v_t ~ N(0,I), E[w_i v_j] = 0, each uncorrelated with any other time lags
    Goal:
        observe {y}, estimate {x}, sigma_v^2
        - assume Q, R invertible
        - use coordinate descent
            - x|sigma_v => least-squares
            - sigma_v|x => (also closed-form)

    http://mlg.eng.cam.ac.uk/zoubin/course04/tr-96-2.pdf
    Ghahramani Z. and Hinton G., "Parameter estimation for linear dynamical systems", 1996.
    partial likelihood (avoid specifying distribution for initial state) version of log p({x},{y}) =
        - sum_{t=2,T} 1/2 (x_t - Ax_{t-1})' 1/sigma_q^2 Q^-1 (x_t - Ax_{t-1})
        - (T-1)/2 log |sigma_q^2 Q|
        - sum_{t=1,T} 1/2 (y_t - Cx_t)' R^-1 (y_t - Cx_t)
        - (T-1)/2 log |R|
    
    finding sigma_q:
    \partial log pmle / \partial sigma_q^-2 =
        (T-1)/2 (dimQ) sigma_q^2 - sum_{t=2,T} 1/2 (x_t - Ax_{t-1})' Q^-1 (x_t - Ax_{t-1})
    since
        log |sigma_v^2 Q|
        = log sigma_v^(2 dimQ) |Q|  where Q = (dimQ, dimQ)
        = dimQ log sigma_v^2 + log |Q|
    and \partial log |Q|^-1 /\partial Q^-1 = Q
    and \partial log sigma^-1 /\partial sigma^-1 = 1/sigma^-1 = sigma        
    
    finding x:
    using (x[2:T] selecting columns = x[:,1:]) for 'x - Ax loss':
    sum_{t=2,T} 1/2 (x_t - Ax_{t-1})' Q^-1 (x_t - Ax_{t-1})
        = ||Q^-1/2(x[2:T] - Ax[1:T-1])||_F^2
        = ||Q^-1/2(x I_T[:,1:] - Ax I_T[1:T-1])||_F^2
        = ||(I_T[1:,:] kron Q^-1/2 I_n)vec(x) - (I_T[:-1,:] kron Q^-1/2 A)vec(x)||_2^2
        = ||(L_a-R_a)vec(x)||_2^2
    using identity (B' kron A)vec(X) = vec(AXB), vec([a_11 a_12 a_21 a_22]) = [a_11 a_21 a_12 a_22] col-major
    and for 'y - Cx loss':
    sum_{t=1,T} 1/2 (y_t - Cx_t)' R^-1 (y_t - Cx_t)
        = ||(I_T kron R^-1/2 C)vec(x) - vec(R^-1/2 Y)||_2^2
        = ||L_y vec(x) - b_y||_2^2
    so total loss = ||[ (L_a-R_a)/sigma_v \\ L_y ]vec(x) - [0 \\ b_y]||_2^2
    :param A: n, n
    :param Q: n, n  | structure of process covariance (minus the scaling factor we want)
    :param C: m, n
    :param R: m, m
    :param y: m, T  | noisy observations
    :param sigma_0: initial value of sigma
    :param x0: initial value of x, (n, T)
    :param n_iter:
    :return: 
        sigma_q:
        x: n, T
    """
    Qi = np.linalg.inv(Q) # Q^-1
    Qih = np.linalg.cholesky(Qi).T  # Q^-1/2 s.t. Q^-1/2T Q^-1/2 = Q^-1
    Ri = np.linalg.inv(R)  # R^-1
    Rih = np.linalg.cholesky(Ri).T  # R^-1/2
    n = A.shape[0]
    T = y.shape[1]
    # calculate quantities for x step
    L_a = sparse.kron(
        sparse.eye(T, T, format='lil')[1:, :],
        sparse.lil_matrix(Qih),
        format='csr'
    )
    R_a = sparse.kron(
        sparse.eye(T, T, format='lil')[:-1, :],
        sparse.lil_matrix(Qih.dot(A)),
        format='csr'
    )
    L_y = sparse.kron(
        sparse.eye(T, T, format='csr'),
        sparse.csr_matrix(Rih.dot(C)),
    )
    b_y = Rih.dot(y).T.ravel()  # vec is col-major
    b = np.hstack((np.zeros((n * (T - 1))), b_y))

    def x_given_sigma(sigma_q, x0=()):
        x0 = x0.ravel(order='F') if len(x0) > 0 else None
        Ap = sparse.vstack(((L_a - R_a) / sigma_q, L_y))
        res = la.lsqr(Ap, b, x0=x0)
        x = np.ascontiguousarray(res[0].reshape((n, T), order='F'))
        return x

    def sigma_given_x(x):
        x_err = x[:, 1:] - A.dot(x[:, :-1])
        weighted_err = (x_err * Qi.dot(x_err)).sum()
        sigma_sq = weighted_err/(n * (T-1))
        return np.sqrt(sigma_sq)

    def f_fcn(x, sigma):
        x_err = x[:, 1:] - A.dot(x[:, :-1])
        a_loss = 0.5 * (x_err * Qi.dot(x_err)).sum() / (sigma ** 2)
        y_err = y - C.dot(x)
        y_loss = 0.5 * (y_err * Ri.dot(y_err)).sum()
        fx = a_loss + y_loss + n * (T-1)/2 * 2 * np.log(sigma)
        return fx

    sigma_q = sigma_0
    x = x0 if len(x0) > 0 else ()   # n, T
    if len(x) > 0:
        print(f_fcn(x, sigma_q))
    for i in range(n_iter):
        x = x_given_sigma(sigma_q, x0=x)
        print(f_fcn(x, sigma_q))
        sigma_q = sigma_given_x(x)
        print(f_fcn(x, sigma_q), sigma_q)
    return sigma_q, x


def stacked_array2list_like(y_list, x):
    """
    :param y_list: list[i] = (m, t_i)
    :param x: (n, sum_i t_i)
    :return: x unstacked into list similar to y_list
    """
    t_csum = np.cumsum([0] + [yi.shape[1] for yi in y_list])
    x_list = [x[:, t_csum[i]:t_csum[i+1]] for i in range(len(y_list))]
    return x_list


def _numerical_x_given_sigma(sigma_q, shape, f_fcn):
    f_obj = lambda x: f_fcn(x.reshape(shape), sigma_q)
    x0 = np.zeros((shape[0]*shape[1],))
    res = opt.minimize(f_obj, x0)
    x_hat = res.x.reshape(shape)
    return x_hat


def _numerical_sigma_given_x(x, f_fcn):
    f_obj = lambda sigma_arr: f_fcn(x, sigma_arr[0])
    x0 = np.array([0.1])
    res = opt.minimize(f_obj, x0)
    sigma_hat = res.x[0]
    return sigma_hat


def _generate_trajectory(x0, sigma_v, sigma_x, dt, n_steps):
    A, Q, C, R = make_ss_matrices(sigma_x, dt)[:-1]
    Q_scaled = sigma_v**2 * Q
    Qh, Rh = np.linalg.cholesky(Q_scaled), np.linalg.cholesky(R)
    m, n = C.shape
    x = np.empty((n, n_steps), dtype=np.float)
    y = np.empty((m, n_steps), dtype=np.float)
    x[:, 0] = x0
    y[:, 0] = C.dot(x0) + Rh.dot(np.random.randn(m))
    for i in range(1, n_steps):
        x[:, i] = A.dot(x[:, i-1]) + Qh.dot(np.random.randn(n))
        y[:, i] = C.dot(x[:, i]) + Rh.dot(np.random.randn(m))
    return x, y


def main_check_trajectory():
    seed = 93473
    np.random.seed(seed)
    sigma_x, sigma_v = 0.1, 0.05
    dt = 0.1
    n_steps = 1000
    x0 = np.array([0., 0, 1.1, .7])
    x, y = _generate_trajectory(x0, sigma_v, sigma_x, dt, n_steps)

    # now try to estimate
    A, Q, C, R, sigma_0 = make_ss_matrices(sigma_x, dt)
    sigma_v_hat, x_hat = estimate_x_sigmaq_coord_descent(A, Q, C, R, y, sigma_0, n_iter=10)
    # sigma_v_hat /= dt  # seems to correct? for dt=0.1 case, smaller dt underestimate more
    print('---')
    print('sigma_v: {:.4f} | sigma_v_hat: {:.4f}'.format(sigma_v, sigma_v_hat))
    print('|sigma_v - sigma_v_hat| = {:.6f}'.format(np.abs(sigma_v - sigma_v_hat)))
    print('x error = {:.6f}'.format(0.5 * ((x - x_hat)**2).sum()))


def main_check_trajectory_blocks():
    seed = 93473
    np.random.seed(seed)
    sigma_x, sigma_v = 0.1, 0.08
    dt = 0.1
    n_steps = 1000
    x0 = np.array([0., 0, 1.1, .7])
    n_blocks = 10
    x_blocks, y_blocks = [], []
    for i in range(n_blocks):
        x_, y_ = _generate_trajectory(x0, sigma_v, sigma_x, dt, n_steps)
        x_blocks.append(x_)
        y_blocks.append(y_)
    x = np.hstack(x_blocks)
    A, Q, C, R, sigma_0 = make_ss_matrices(sigma_x, dt)
    sigma_v_hat, x_hat = estimate_blocks_x_sigmaq_coord_descent(
        A, Q, C, R, y_blocks, sigma_0, n_iter=10)
    print('---')
    print('sigma_v: {:.4f} | sigma_v_hat: {:.4f}'.format(sigma_v, sigma_v_hat))
    print('|sigma_v - sigma_v_hat| = {:.6f}'.format(np.abs(sigma_v - sigma_v_hat)))
    print('x error = {:.6f}'.format(0.5 * ((x - x_hat)**2).sum()))

    sigma_v_hat, x_hat = estimate_blocks_lo_x_sigmaq_coord_descent(
        A, Q, C, R, y_blocks, sigma_0, n_iter=10)
    print('---')
    print('sigma_v: {:.4f} | sigma_v_hat: {:.4f}'.format(sigma_v, sigma_v_hat))
    print('|sigma_v - sigma_v_hat| = {:.6f}'.format(np.abs(sigma_v - sigma_v_hat)))
    print('x error = {:.6f}'.format(0.5 * ((x - x_hat)**2).sum()))


if __name__ == '__main__':
    # main_check_trajectory()
    main_check_trajectory_blocks()




