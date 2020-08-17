import numpy as np
import test_problems.risk_distances as rt
import test_problems.perp_dist as perp
from utils.tt_dataset import AgentType
import utils.general as ge


# < 0 to ensure no real agent_id is taken
R_NO_VIC = -1
R_SENTINEL = -2


def dataset_filter_minimum_obs(dataset, tau_frames=3):
    """
    :param dataset: TrajectoryTypeDataset: exist [frame_id(index) agent_id]
    :param tau_frames: minimum number of frames
    :return:
    """
    for df_info in dataset.df_list:
        df = df_info.df
        agent_ids = df['agent_id'].unique()
        valid_agent_ids = set()
        for agent_id in agent_ids:
            agent_df = df.loc[df.agent_id == agent_id]
            if agent_df.shape[0] >= tau_frames:
                valid_agent_ids.add(agent_id)
        df_info.df = df.loc[df.agent_id.isin(valid_agent_ids)]


def dataset_filter_stationary_vic(dataset, dt, tau_s=0.5):
    """
    :param dataset:
    :param dt:
    :param tau_s: speed cutoff for stationary
    :return:
    """
    for df_info in dataset.df_list:
        df = df_info.df
        agent_ids = df.loc[df.type_id == AgentType.vic, 'agent_id'].unique()
        valid_agent_ids = set(df.loc[df.type_id != AgentType.vic, 'agent_id'])
        for agent_id in agent_ids:
            agent_df = df.loc[df.agent_id == agent_id]
            xy = agent_df[['x', 'y']].values
            s = np.linalg.norm(xy[1:] - xy[:-1], axis=1) / dt
            if np.all(s >= tau_s):
                valid_agent_ids.add(agent_id)
        df_info.df = df.loc[df.agent_id.isin(valid_agent_ids)]


def dataset_estimate_r(dataset, dt, is_rm_unknown=True, tau_frames=3):
    """
    Modify dataset and add 'r' estimates and
    if for given ped it is always known, 'is_r_all'
    - ignore pedestrians that are not continuously tracked
    - or if number of frames is too few
    - except the last frame - since we need difference to (roughly) estimate velocity
    Estimate r indices based on all vic occurring in pedestrian's entire track
    - ordered by agent_id
    Consistent with ~is_obs estimation, cannot estimate final timestep \hat{v}
    - so include it here, drop it there
    Note:
    - this sets r = indices of vic_xy extracted from ped's frames
      instead of actual vic's agent_id
    :param dataset: TrajectoryTypeDataset
    :param dt:
    :param is_rm_unknown: remove pedestrians who contain unknown r
    :param tau_frames: minimum number of frames
    :return:
    """
    for df_info in dataset.df_list:
        df = df_info.df
        df['r'] = R_SENTINEL
        df['is_r_all'] = False
        df.loc[df.type_id != AgentType.ped, 'is_r_all'] = True
        ped_ids = df.loc[df.type_id == AgentType.ped, 'agent_id'].unique()
        for ped_id in ped_ids:
            ped_df = df.loc[df.agent_id == ped_id]
            frames = ped_df.index.unique()
            is_cts = frames.size == frames[-1] - frames[0] + 1
            if not is_cts or frames.size < tau_frames:
                continue
            seq_df = df.loc[frames]
            seq_df = seq_df.loc[(seq_df.agent_id == ped_id) |
                                (seq_df.type_id == AgentType.vic)]
            ped_xy, vic_xy = dataset.build_nan_df(seq_df, frames[0], frames.size)
            r = estimate_r_v1(ped_xy, vic_xy, dt)
            mask_ped = df.agent_id == ped_id
            mask_frames = mask_ped & df.index.isin(frames[:-1])
            df.loc[mask_frames, 'r'] = r
            df.loc[mask_ped, 'is_r_all'] = (r != R_SENTINEL).all()
        if is_rm_unknown:
            df_info.df = df.loc[df.is_r_all]


def dataset_set_is_obs_frames(dataset, dt, tau_dist, tau_frames=3):
    # R_NO_VIC => is_obs may not hold if observed frames not consecutive
    for df_info in dataset.df_list:
        df = df_info.df
        df['is_obs'] = False
        df.loc[df.type_id != AgentType.ped, 'is_obs'] = True
        ped_ids = df.loc[df.type_id == AgentType.ped, 'agent_id'].unique()
        for ped_id in ped_ids:
            ped_df = df.loc[df.agent_id == ped_id]
            frames = ped_df.index.unique()
            is_cts = frames.size == frames[-1] - frames[0] + 1
            if not is_cts or frames.size < tau_frames:
                continue
            seq_df = df.loc[frames]
            seq_df = seq_df.loc[(seq_df.agent_id == ped_id) |
                                (seq_df.type_id == AgentType.vic)]
            ped_xy, vic_xy = dataset.build_nan_df(seq_df, frames[0], frames.size)
            # frame_inds = find_little_interaction_frames_v0(ped_xy[:, 0, :], vic_xy, dt)
            # frame_inds = find_little_interaction_frames_v1(ped_xy[:, 0, :], vic_xy, tau_dist)
            frame_inds = find_little_interaction_frames_v2(ped_xy[:, 0, :], vic_xy, tau_dist)
            mask = (df.agent_id == ped_id) & \
                   (df.index.isin(frames[frame_inds]))
            df.loc[mask, 'is_obs'] = True
        print('df unobs total: ', (~df['is_obs'].values).sum())


def dataset_set_r_and_is_obs_frames(dataset, dt, tau_dist, tau_frames=3,
                                    is_rm_unknown=True):
    # R_t =
    # {} => is_obs
    # {i} => not is_obs
    # for ped i, if any |R_t| > 1, => remove
    for df_info in dataset.df_list:
        df = df_info.df
        df['r'] = R_SENTINEL
        df['is_r_all'] = False
        df.loc[df.type_id != AgentType.ped, 'is_r_all'] = True
        df['is_obs'] = False
        df.loc[df.type_id != AgentType.ped, 'is_obs'] = True
        ped_ids = df.loc[df.type_id == AgentType.ped, 'agent_id'].unique()

        for ped_id in ped_ids:
            ped_df = df.loc[df.agent_id == ped_id]
            frames = ped_df.index.unique()
            is_cts = frames.size == frames[-1] - frames[0] + 1
            if not is_cts or frames.size < tau_frames:
                continue
            seq_df = df.loc[frames]
            seq_df = seq_df.loc[(seq_df.agent_id == ped_id) |
                                (seq_df.type_id == AgentType.vic)]
            ped_xy, vic_xy = dataset.build_nan_df(seq_df, frames[0], frames.size)
            rt_set_list, rt_size, r = set_r_and_is_obs_v0(
                ped_xy[:, 0, :], vic_xy, tau_dist)
            _ = (r[:-1] != R_SENTINEL).all()
            mask = (df.agent_id == ped_id) & \
                   (df.index.isin(frames))
            df.loc[mask & (df.index.isin(frames[:-1])), 'r'] = r[:-1]
            df.loc[mask, 'is_r_all'] = (r[:-1] != R_SENTINEL).all()
            df.loc[mask & (df.index.isin(frames[:-1][r[:-1] == R_NO_VIC])), 'is_obs'] = True

        if is_rm_unknown:
            df_info.df = df.loc[df.is_r_all]
        print('df unobs total: ', (~df['is_obs'].values).sum())


def estimate_r(ped_xy, vic_xy, dt):
    """
    No vic in frame / strictly dominating vic => set r to this vic's index
    Strictly dominating defined as
    - dsq < any other vic's dsq
    - tau < any other vic's tau
    :param ped_xy: n_frames, n_ped, 2
    :param vic_xy: n_frames, n_vic, 2
    :param dt: 
    :return: n_frames-1, n_ped | index of dominating vic for frame,
      i = R_NO_VIC if no vic
      i = R_SENTINEL if not estimated (ie no dominating vic exists, or ped = nan in frame i/i+1)
    """
    n_frames, n_ped = ped_xy.shape[:2]
    n_vic = vic_xy.shape[1]
    r = np.zeros((n_frames - 1, n_ped), dtype=np.int) + R_SENTINEL
    for i in range(n_ped):
        for j in range(n_frames-1):
            ped_pv = np.concatenate([
                ped_xy[[j], i, :], (ped_xy[[j + 1], i, :] - ped_xy[[j], i, :]) / dt], axis=1)
            vic_pv = np.concatenate([
                vic_xy[j, :, :], (vic_xy[j + 1, ...] - vic_xy[j, ...]) / dt], axis=1)
            is_vic_nan = np.isnan(vic_pv).any(axis=1)
            if is_vic_nan.sum() == n_vic:
                r[j, i] = R_NO_VIC
                continue
            if np.isnan(ped_pv).any():
                continue
            dsq_tau = rt.dist2rt_v1(ped_pv, vic_pv[~is_vic_nan, :])  # (n_ped=1, n_vic, 2)
            dsq_tau = dsq_tau[0, ...]
            ind_dominating = find_dominating(-dsq_tau, sentinel=R_SENTINEL)
            ind_dominating = np.arange(n_vic)[~is_vic_nan][ind_dominating]
            r[j, i] = ind_dominating
    return r


def estimate_r_v1(ped_xy, vic_xy, dt):
    """
    No vic in frame / strictly dominating vic => set r to this vic's index
    Strictly dominating defined as
    - dsq < any other vic's dsq
    - tau < any other vic's tau
    :param ped_xy: n_frames, n_ped, 2
    :param vic_xy: n_frames, n_vic, 2
    :param dt:
    :return: n_frames-1, n_ped | index of dominating vic for frame,
      i = R_NO_VIC if no vic
      i = R_SENTINEL if not estimated (ie no dominating vic exists, or ped = nan in frame i/i+1)
    """
    n_frames, n_ped = ped_xy.shape[:2]
    n_vic = vic_xy.shape[1]
    vic_pv_all = ge.xy2pv(vic_xy[:-1, ...])
    r = np.zeros((n_frames - 1, n_ped), dtype=np.int) + R_SENTINEL
    for i in range(n_ped):
        for j in range(n_frames-1):
            ped_pv = np.concatenate([
                ped_xy[[j], i, :], (ped_xy[[j + 1], i, :] - ped_xy[[j], i, :]) / dt], axis=1)
            # vic_pv = np.concatenate([
            #     vic_xy[j, :, :], (vic_xy[j + 1, ...] - vic_xy[j, ...]) / dt], axis=1)
            vic_pv = vic_pv_all[j, ...]
            is_vic_nan = np.isnan(vic_pv).any(axis=1)
            if is_vic_nan.sum() == n_vic:
                r[j, i] = R_NO_VIC
                continue
            if np.isnan(ped_pv).any():
                continue
            dsq_tau = rt.dist2rt_v1(ped_pv, vic_pv[~is_vic_nan, :])  # (n_ped=1, n_vic, 2)
            dsq_tau = dsq_tau[0, ...]
            ind_dominating = find_dominating(-dsq_tau, sentinel=R_SENTINEL)
            ind_dominating = np.arange(n_vic)[~is_vic_nan][ind_dominating]
            r[j, i] = ind_dominating
    return r


def estimate_r_v2(ped_xy, vic_xy, dt):
    """
    No vic in frame / 1 vic => set r to this vic's index
    Strictly dominating defined as
    - dsq < any other vic's dsq
    - tau < any other vic's tau
    :param ped_xy: n_frames, n_ped, 2
    :param vic_xy: n_frames, n_vic, 2
    :param dt:
    :return: n_frames-1, n_ped | index of dominating vic for frame,
      i = R_NO_VIC if no vic
      i = R_SENTINEL if not estimated (ie no dominating vic exists, or ped = nan in frame i/i+1)
    """
    n_frames, n_ped = ped_xy.shape[:2]
    n_vic = vic_xy.shape[1]
    vic_pv_all = ge.xy2pv(vic_xy[:-1, ...])
    r = np.zeros((n_frames - 1, n_ped), dtype=np.int) + R_SENTINEL
    for i in range(n_ped):
        for j in range(n_frames-1):
            ped_pv = np.concatenate([
                ped_xy[[j], i, :], (ped_xy[[j + 1], i, :] - ped_xy[[j], i, :]) / dt], axis=1)
            # vic_pv = np.concatenate([
            #     vic_xy[j, :, :], (vic_xy[j + 1, ...] - vic_xy[j, ...]) / dt], axis=1)
            vic_pv = vic_pv_all[j, ...]
            is_vic_nan = np.isnan(vic_pv).any(axis=1)
            if is_vic_nan.sum() == n_vic:
                r[j, i] = R_NO_VIC
                continue
            if np.isnan(ped_pv).any():
                continue
            if is_vic_nan.sum() == n_vic-1:
                r[j, i] = np.arange(n_vic)[~is_vic_nan][0]
    return r


def estimate_v(ped_xy, vic_xy, dt):
    """
    Estimate 'desired' velocity v_t for frames with interaction
    - based on frames with little interaction
    1) find consecutive xy to estimate v, for all peds
    - calculate sigma without sigma_v scaling factor
    2) [not in this function] calculate sigma_v based on these noiseless v estimates
    - using consecutive v with sigma <= 0
    - apply sigma_v to calculate sigma
    :param ped_xy: n_frames, n_ped, 2
    :param vic_xy: n_frames, n_vic, 2
    - if vic not observed, for those frames can set nan to ignore
    :return: 
      v: n_frames-1, n_ped, 2 | estimated velocities
      sigma: n_frames-1, n_ped | estimated standard deviation, < 0 for those exact
      - without sigma_v scaling factor
      - exact for velocities calculated entirely from differences
    """
    n_frames, n_ped = ped_xy.shape[:2]
    v = np.zeros((n_frames-1, n_ped, 2))
    sigma = np.zeros((n_frames-1, n_ped)) - 1.
    for i in range(n_ped):
        frame_inds = find_little_interaction_frames_v0(ped_xy[:, i, :], vic_xy, dt)
        if len(frame_inds) == 0:
            continue
        v[:, i, :], sigma[:, i] = estimate_v_ped_i(ped_xy[:, i, :], frame_inds, dt)
    return v, sigma


def estimate_v_ped_i(ped_xy, frame_inds, dt):
    """
    
    :param ped_xy: n_frames, 2
    :param frame_inds: inds of ped_xy such that xy considered observed for i,i+1
    - assume len > 0
    - find nearest observed ind, before and after (3 cases since at least one exists) 
    :param dt: 
    :return: 
        v: 
        sigma: without sigma_v scaling factor | <= 0 if observed
    """
    n_frames = ped_xy.shape[0]
    v = np.zeros((n_frames - 1, 2))
    sigma = np.zeros((n_frames - 1, )) - 1.

    mask_consec = np.full((n_frames-1,), False)
    mask_consec[frame_inds] = True
    v_hat = (ped_xy[1:, :] - ped_xy[:-1, :])/dt
    v[mask_consec, :] = v_hat[mask_consec, :]

    nearest_ind_before = np.repeat(
        np.hstack((-1, frame_inds)),
        repeats=np.hstack((frame_inds, n_frames-1))-np.hstack((0, frame_inds))
    )
    nearest_ind_after = np.repeat(
        np.hstack((frame_inds, -1)),
        repeats=np.hstack((frame_inds, n_frames-1))-np.hstack((0, frame_inds))
    )
    # after-only
    m1 = nearest_ind_before == -1
    if np.any(m1):
        inds = nearest_ind_after[m1]
        v[m1, :] = v_hat[inds, :]
        l_after = inds - np.arange(n_frames-1)[m1]
        sigma[m1] = l_after ** 2
    # before-only
    m2 = nearest_ind_after == -1
    if np.any(m2):
        inds = nearest_ind_before[m2]
        v[m2, :] = v_hat[inds, :]
        l_before = np.arange(n_frames-1)[m2] - inds
        sigma[m2] = l_before ** 2
    # both (but not consecutive)
    m3 = ~(m1 | m2 | mask_consec)
    if np.any(m3):
        inds_before = nearest_ind_before[m3]
        inds_after = nearest_ind_after[m3]
        l_before = (np.arange(n_frames-1)[m3] - inds_before)**2
        l_after = (inds_after - np.arange(n_frames-1)[m3])**2
        v[m3, :] = ((l_before*v_hat[inds_before, :].T + l_after*v_hat[inds_after, :].T)
                    /(l_before + l_after)).T
        sigma[m3] = 1/(1./l_before + 1./l_after)
    return v, sigma


def find_little_interaction_frames_v0(ped_xy, vic_xy, dt):
    """
    little interaction if either of
    - rt risk threshold met: for each vic, ped not on collision course, or vic is stationary
    - perpendicular distance: even if collision course, distance is sufficiently far
      - this relates to finite (perp) distance m(x)=u maps encode
    :param ped_xy: n_frames, 2
    :param vic_xy: n_frames, n_vic, 2
    - frames with nan ignored
    :param dt: 
    :return: frame_inds: inds of ped_xy *velocity* counted as observed, in sorted order
    - ind i included if i,i+1 both observed or i has no vic
    """
    tau_dist = 5  # meters away from any vic's last-velocity line
    tau_v = 0.5  # threshold for stationary vics
    M = 20  # threshold for d, tau - to check that ped is on collision course
    frame_inds = []
    n_frames = ped_xy.shape[0]
    no_vic_frames = np.arange(n_frames)[
        np.isnan(vic_xy).any(axis=2).all(axis=1)
    ]
    # only for frames where i, i+1 not nan
    is_ped_valid = ~np.isnan(ped_xy[:-1, 0]) & ~np.isnan(ped_xy[1:, 0])
    for i in np.arange(n_frames-1)[is_ped_valid]:
        if i in no_vic_frames:
            continue
        ped_pv = np.concatenate([ped_xy[[i], :], (ped_xy[[i+1], :] - ped_xy[[i], :])/dt], axis=1)
        vic_pv = np.concatenate([vic_xy[i, :, :], (vic_xy[i+1, ...] - vic_xy[i, ...])/dt], axis=1)
        is_vic_nan = np.isnan(vic_pv).any(axis=1)
        vic_pv = vic_pv[~is_vic_nan, :]
        dsq_tau = rt.dist2rt_v1(ped_pv, vic_pv)  # (n_ped=1, n_vic, 2)
        dsq_tau = dsq_tau[0, ...]
        dsq_tau[:, 0] = np.sqrt(dsq_tau[:, 0])
        is_intersect = np.all(dsq_tau < M, axis=1)
        is_stationary = (vic_pv[:, 2:]**2).sum(axis=1) < tau_v**2
        mask = is_intersect & ~is_stationary
        if np.sum(mask) == 0:
            frame_inds.append(i)
            continue
        pe_dist = perp.pe_dist_v0(ped_pv[:, :2], vic_pv[mask, :])[0]
        if np.all(pe_dist > tau_dist):
            frame_inds.append(i)
    consec_inds = np.sort(np.hstack((frame_inds, no_vic_frames)).astype(np.int))
    consec_inds = ge.select_consecutive_forward(consec_inds)
    # add any removed no_vic frames back in
    consec_inds = np.unique(np.hstack((consec_inds, no_vic_frames))).astype(np.int)
    return consec_inds


def find_little_interaction_frames_v1(ped_xy, vic_xy, tau_dist, is_signed=False):
    # Frame i has little interaction for ped if either:
    # 1) ped_xy st. perpendicular distance > tau_dist
    # 2) no vic
    # - no dependence on ped velocity
    # - for tau_dist use l from MRX_GRID = (l, m), with [0, ..., l].size = m
    # - assume pedestrian observed entirely in given sequence
    n_frames = ped_xy.shape[0]
    no_vic_frames = np.arange(n_frames)[
        np.isnan(vic_xy).any(axis=2).all(axis=1)
    ]
    frame_inds = []
    for i in np.arange(n_frames-1):
        if i in no_vic_frames:
            continue
        vic_pv = np.concatenate([vic_xy[i, :, :], vic_xy[i+1, ...] - vic_xy[i, ...]], axis=1)
        vic_pv = vic_pv[~np.isnan(vic_pv).any(axis=1), :]
        pe_dist, pa_dist = perp.pe_dist_v0(ped_xy[[i], :], vic_pv)
        if not is_signed and np.all(pe_dist > tau_dist):
            frame_inds.append(i)
        elif np.all((pe_dist > tau_dist) | (pa_dist < 0)):
            frame_inds.append(i)
    consec_inds = np.sort(np.hstack((frame_inds, no_vic_frames)).astype(np.int))
    consec_inds = ge.select_consecutive_forward(consec_inds)
    # add any removed no_vic frames back in
    consec_inds = np.unique(np.hstack((consec_inds, no_vic_frames))).astype(np.int)
    return consec_inds


def find_little_interaction_frames_v2(ped_xy, vic_xy, tau_dist, vic_n_roll=2):
    # Frame i has little interaction for ped if either:
    # 1) ped_xy st. perpendicular distance (< -tau_dist), (> 0),
    #    or parallel distance < 0
    # 2) no vic
    # - weak dependence on ped velocity - only direction needed: use average v
    # - for tau_dist use l from MRX_GRID = (l, m), with [0, ..., l].size = m
    # - assume pedestrian observed entirely in given sequence
    n_frames = ped_xy.shape[0]
    if vic_xy.shape[1] == 0:
        return np.arange(n_frames).astype(np.int)
    no_vic_frames = np.arange(n_frames)[
        np.isnan(vic_xy).any(axis=2).all(axis=1)
    ]
    frame_inds = []
    ped_pv = np.zeros((n_frames-1, 4))
    ped_pv[:, :2] = ped_xy[:-1]
    ped_pv[:, 2:] = (ped_xy[1:] - ped_xy[:-1]).mean(axis=0)
    vic_pv = ge.xy2pv(vic_xy[:-1, ...], n_roll=vic_n_roll)
    for i in np.arange(n_frames-1):
        if i in no_vic_frames:
            continue
        vic_pv_i = vic_pv[i, :, :]
        vic_pv_i = vic_pv_i[~np.isnan(vic_pv_i).any(axis=1), :]
        pe_dist, pa_dist = perp.signed_pe_dist_v0(ped_pv[[i], :], vic_pv_i)  # 1, k
        #
        is_far = pe_dist < -tau_dist
        is_crossed = pe_dist > 0
        is_behind = pa_dist < 0
        if np.all(is_far | is_crossed | is_behind):
            frame_inds.append(i)
        #
        # A_vic2world is np.array([yvn, [-yvn[1], yvn[0]]]).T
        # ( 0 -1
        #   1  0 ) * y_v_normalized * x_\perp = x_\perp_world
        # left term = latitude axis in world frame
        # yvn = vic_pv_i[:, 2:].T / np.linalg.norm(vic_pv_i[:, 2:], axis=-1)  # 2, k
        # x_perp_world = np.array([-yvn[1], yvn[0]]) * np.abs(pe_dist)
        # is_crossed = ped_pv[[i], 2:].dot(x_perp_world) >= 0
        # # is_crossed = pe_dist > 0
        # is_far = np.abs(pe_dist) > tau_dist
        # is_behind = pa_dist < 0
        # if np.all(is_far | is_crossed | is_behind):
        #     frame_inds.append(i)
        #
        #
        # is_far = np.abs(pe_dist) > tau_dist
        # is_behind = pa_dist < 0
        # if np.all(is_far | is_behind):
        #     frame_inds.append(i)
        #
    consec_inds = np.sort(np.hstack((frame_inds, no_vic_frames)).astype(np.int))
    consec_inds = ge.select_consecutive_forward(consec_inds)
    consec_inds = np.unique(np.hstack((consec_inds, no_vic_frames))).astype(np.int)
    return consec_inds


def set_r_and_is_obs_v0(ped_xy, vic_xy, tau_dist, vic_n_roll=2):
    n_frames = ped_xy.shape[0]
    rt_set_list = [[] for _ in range(n_frames)]
    rt_size = np.zeros(n_frames)
    rt_size[:] = -1
    r = np.zeros(n_frames)
    r[:] = R_SENTINEL
    if vic_xy.shape[1] == 0:
        rt_set_list = [[] for _ in range(n_frames)]
        rt_size[:] = 0
        r[:] = R_NO_VIC
        return rt_set_list, rt_size, r
    no_vic_frames = np.arange(n_frames)[
        np.isnan(vic_xy).any(axis=2).all(axis=1)
    ]
    ped_pv = np.zeros((n_frames-1, 4))
    ped_pv[:, :2] = ped_xy[:-1]
    # ped_pv[:, 2:] = (ped_xy[1:] - ped_xy[:-1]).mean(axis=0)
    ped_pv = ge.xy2pv(ped_xy[:-1, np.newaxis, :], n_roll=20)[:, 0, :]
    vic_pv = ge.xy2pv(vic_xy[:-1, ...], n_roll=vic_n_roll)
    vic_id = np.arange(vic_pv.shape[1])
    for i in np.arange(n_frames-1):
        if i in no_vic_frames:
            rt_set_list[i] = []
            rt_size[i] = 0
            r[i] = R_NO_VIC
            continue
        vic_pv_i = vic_pv[i, :, :]
        is_valid_mask = ~np.isnan(vic_pv_i).any(axis=1)
        vic_pv_i = vic_pv_i[is_valid_mask, :]
        pe_dist, pa_dist = perp.signed_pe_dist_v0(ped_pv[[i], :], vic_pv_i)  # 1, k
        # A_vic2world is np.array([yvn, [-yvn[1], yvn[0]]]).T
        # ( 0 -1
        #   1  0 ) * y_v_normalized * x_\perp = x_\perp_world
        # left term = latitude axis in world frame
        yvn = vic_pv_i[:, 2:].T / np.linalg.norm(vic_pv_i[:, 2:], axis=-1)  # 2, k
        x_perp_world = np.array([-yvn[1], yvn[0]]) * np.abs(pe_dist)
        is_crossed = ped_pv[i, 2:].dot(x_perp_world) >= 0
        # is_crossed = pe_dist > 0
        is_far = np.abs(pe_dist[0]) > tau_dist
        is_behind = pa_dist[0] < 0-2
        is_attention = ~(is_far | is_crossed | is_behind)
        rt_set_list[i] = vic_id[is_valid_mask][is_attention]
        rt_size[i] = len(rt_set_list[i])
        if rt_size[i] > 1:
            r[i] = R_SENTINEL
        elif rt_size[i] == 1:
            r[i] = rt_set_list[i][0]
        else:
            r[i] = R_NO_VIC
    return rt_set_list, rt_size, r


def find_dominating(val, sentinel=-999):
    """
    Find index of strictly dominating agent based on values
    - treat nans as minimum value
    :param val: n, m | m values for each of n agents
    :param sentinel: value to return in case of no dominating agent
    :return: 
    """
    v = val.copy()
    v[np.isnan(v)] = -np.inf
    n = val.shape[0]
    i = np.argmax(v[:, 0])
    gt = v[list(set(range(n)) - {i}), :] - v[i, :]
    i = i if (gt < 0).all() else sentinel
    return i


def main_check():
    n_frames = 6
    one_ped = np.array([
        np.arange(n_frames),
        np.arange(n_frames)]).T
    ped_xy = np.concatenate([
        one_ped[:, np.newaxis, :],
        one_ped[:, np.newaxis, :]+10], axis=1)
    dt = 1.
    frame_inds = np.array([0, 3, 4], dtype=int)
    v, sigma = estimate_v_ped_i(ped_xy[:, 0, :], frame_inds, dt)
    print(v)
    print(sigma)


if __name__ == '__main__':
    main_check()







