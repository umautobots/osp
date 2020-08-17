import numpy as np
import ss_model.data_selection as ss
import display.utils as ut
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


DT = 0.1


def display_dataset_qr_per_frame_window(
        dataset, df_id2inds, n_frames, n_skip=1,
        is_draw_r=True):
    """

    :param dataset:
    :param df_id2inds:
    :param n_frames:
    :param n_skip:
    :param is_draw_r:
    :return:
    """
    for df_info in dataset.df_list:
        if df_info.datafile_path not in df_id2inds:
            continue
        df = df_info.df
        frames = df.index.unique()[::n_skip]
        for frame_0 in frames:
            ped_xy, vic_xy = dataset.build_nan_df(df, frame_0, n_frames)
            data_title = ut.format_frame_window_title(
                frame_0, n_frames, df_info.datafile_path)
            ax = display_all(ped_xy, vic_xy, data_title=data_title)
            seq_df = df.loc[frame_0:frame_0 + n_frames - 1]
            display_df_q(ax, seq_df)
            if is_draw_r:
                display_df_r(ax, seq_df)
            plt.show()


def display_dataset_qr_per_agent(dataset, df_id2inds):
    """

    :param dataset:
    :param df_id2inds:
    :return:
    """
    for df_info in dataset.df_list:
        if df_info.datafile_path not in df_id2inds:
            continue
        df_inds = df_id2inds[df_info.datafile_path]
        df = df_info.df
        ped_ids = np.unique(df_inds.agent_ids)
        for ped_id in ped_ids:
            ped_df = df.loc[df.agent_id == ped_id]
            frames = ped_df.index.unique()
            seq_df = df.loc[frames]
            ped_xy, vic_xy = dataset.build_nan_df(seq_df, frames[0], frames.size)
            data_title = ut.format_agent_title(ped_id, df_info.datafile_path)
            ax = display_all(ped_xy, vic_xy, data_title=data_title)
            display_df_q(ax, seq_df, ped_id)
            display_df_r(ax, seq_df, ped_id)
            plt.show()


def display_df_q(ax, df, agent_id=np.nan):
    mask = ~df.q.isna()
    if not np.isnan(agent_id):
        mask = mask & (df.agent_id == agent_id)
    q_df = df.loc[mask]
    xy_q1 = q_df.loc[q_df.q == 1, ['x', 'y']].values
    ax.plot(
        xy_q1[:, 0], xy_q1[:, 1],
        c='green', ls='', marker='o', alpha=0.5,
        markersize=10
    )
    xy_q0 = q_df.loc[q_df.q == 0, ['x', 'y']].values
    ax.plot(
        xy_q0[:, 0], xy_q0[:, 1],
        c='orange', ls='', marker='s', alpha=0.5,
        markersize=10
    )


def display_df_r(ax, df, agent_id=np.nan):
    mask = df.r == ss.R_SENTINEL
    if not np.isnan(agent_id):
        mask = mask & (df.agent_id == agent_id)
    xy = df.loc[mask, ['x', 'y']].values
    ax.plot(
        xy[:, 0], xy[:, 1],
        c='red', ls='', marker='^', alpha=0.5,
        markersize=10
    )


def display_all(ped_xy, vic_xy, data_title=''):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    n_frames, n_ped = ped_xy.shape[:2]
    for i in range(n_ped):
        ax.plot(
            ped_xy[:, i, 0], ped_xy[:, i, 1],
            c='black', ls='', marker='+', alpha=0.8
        )
        last_ind = np.arange(n_frames)[~np.isnan(ped_xy[:, i, 0])][-1]
        ax.plot(
            ped_xy[last_ind, i, 0], ped_xy[last_ind, i, 1],
            c='magenta', marker='o', alpha=0.5
        )
    n_vic = vic_xy.shape[1]
    for i in range(n_vic):
        ax.plot(
            vic_xy[:, i, 0], vic_xy[:, i, 1],
            c='orange', alpha=0.5
        )
        last_ind = np.arange(n_frames)[~np.isnan(vic_xy[:, i, 0])][-1]
        ax.plot(
            vic_xy[last_ind, i, 0], vic_xy[last_ind, i, 1],
            c='orange', marker='o', alpha=0.5
        )
    ax.set(aspect='equal')
    fig.suptitle(data_title)
    return ax


def display_v_partial(ax, ped_xy, vic_xy):
    """
    For estimated v (sigma > 0), plot one-step prediction from its position
    - to compare to actual velocity
    - at predicted position display plot sigma for v
    :param ax: 
    :param ped_xy: 
    :param vic_xy: 
    :return: 
    """
    sigma_scaling = 5.  # since estimated sigma does not have sigma_v scaling factor
    v, sigma = ss.estimate_v(ped_xy, vic_xy, DT)
    # v_seen = v[sigma[:, 0] <= 0, 0, 0]
    # if v_seen.size > 1:
    #     print(sigma[:, 0] <= 0)
    #     print(v_seen)
    #     print(((v_seen[1:] - v_seen[:-1])**2).mean())
    sigma *= sigma_scaling
    n_ped = ped_xy.shape[1]
    for i in range(n_ped):
        is_est = sigma[:, i] > 0
        if np.sum(is_est) == 0:
            continue
        xy = ped_xy[:-1, ...][is_est, i, :] + v[is_est, i, :] * DT
        sz = sigma[is_est, i]
        ax.scatter(xy[:, 0], xy[:, 1], sz, c='blue', alpha=0.1)


def display_r_is_not_set(ax, ped_xy, vic_xy):
    """
    For displaying data selected as r_t = known
    - "obviously 1 vehicle strictly dominating all others"
    - or no vic present
    - highlight pedestrians for which this does not hold
    :param ax: 
    :param ped_xy: 
    :param vic_xy: 
    :return: 
    """
    r_est = ss.estimate_r(ped_xy, vic_xy, DT)
    n_ped = ped_xy.shape[1]
    for i in range(n_ped):
        is_r_nan = np.isnan(r_est[:, i])
        if np.sum(is_r_nan) == 0:
            continue
        xy = ped_xy[:-1, ...][is_r_nan, i, :]
        ax.scatter(xy[:, 0], xy[:, 1], 20., c='red', alpha=0.05)
