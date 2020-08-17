import numpy as np
import pandas as pd
import warnings


def xy2pv(xy, n_roll=0):
    """

    :param xy: t, k, 2
    :param n_roll: window size to use in rolling mean
    :return:
        pv: t, k, 4
    """
    pv = np.zeros((xy.shape[0], xy.shape[1], 4))
    pv[..., :2] = xy
    if xy.shape[0] > n_roll > 0:
        pv[..., 2:] = rolling_mean_difference(pv[..., :2], n_roll=n_roll)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            pv[..., 2:] = np.nanmean(xy[1:, ...] - xy[:-1, ...], axis=0)
    return pv


def rolling_mean_difference(a, n_roll=2, pad_constant=True):
    """
    Calculate rolling mean, assuming nans if exist, only padding valid values
    :param a: t, k, n | array of values across time for k agents
    :param n_roll: window size to use in rolling mean
    :param pad_constant: fill nans (from roll) with nearest (last/next) valid value
    :return: t, k, n
    """
    diff = 0*a
    for i in range(a.shape[1]):
        df = pd.DataFrame(a[:, i, :])
        rolling_mean = df.diff().rolling(n_roll, min_periods=1).mean()
        if pad_constant:
            rolling_mean = rolling_mean.bfill().ffill()
        diff[:, i, :] = rolling_mean.values
    return diff


def select_consecutive_forward(frames):
    """
    Select frame i if i+1 observed
    :param frames:
    :return:
    """
    if frames.size < 2:
        return frames
    mask = frames[1:] - frames[:-1] == 1
    consecutive = frames[:-1][mask]
    return consecutive


def select_consecutive(frames):
    """
    Select frame i if [i-1, i+1] observed
    :param frames:
    :return:
    """
    if frames.size < 2:
        return frames
    mask = frames[1:] - frames[:-1] == 1
    mask = np.hstack((mask[0], mask[1:] & mask[:-1], mask[-1]))
    consecutive = frames[mask]
    return consecutive


def split_to_consecutive_v0(frames, tau_frames=1):
    """
    [a, a+1, ..., a+na, b, b+1, ... ] ->
    [a, a+1, ..., a+na], [b, ... b+nb], ...
    :param frames: n, | integers, sorted ascending
    :param tau_frames: minimum number of frames in partition
    :return:
    """
    if frames.size < 2:
        return frames,
    split_inds = [0]
    for i in range(1, frames.size):
        if frames[i] != frames[i-1] + 1:
            split_inds.append(i)
    split_inds.append(frames.size)
    ptns = [frames[split_inds[i]:split_inds[i+1]]
            for i in range(len(split_inds)-1)
            if split_inds[i+1]-split_inds[i] >= tau_frames]
    return ptns


def find_box_contained_agents(xy, xlim, ylim):
    """
    Find inds for which of n agents contained in given limits (inclusive)
    :param xy: t, n, 2
    :param xlim: 2, | min, max
    :param ylim: 2, | min, max
    :return: n,
    """
    n = xy.shape[1]
    mask = np.ones(n).astype(np.bool)
    if len(xlim) > 0:
        mask &= is_series_contained(xy[..., 0], xlim)
    if len(ylim) > 0:
        mask &= is_series_contained(xy[..., 1], ylim)
    return np.arange(n)[mask]


def is_series_contained(x, lim):
    """
    :param x: t, n
    :param lim: 2,
    :return:
    """
    return (lim[0] <= x).all(axis=0) & \
           (x <= lim[1]).all(axis=0)


def select_moving(x, tau=0.5):
    """

    :param x: t, k, 2
    :param tau:
    :return:
    """
    start = x[0, ...]  # k, 2
    is_stationary = np.linalg.norm(x - start, axis=-1) < tau
    return x[:, ~(is_stationary.all(axis=0)), :]


def find_char_inds(s, c):
    return np.arange(len(s))[np.array(list(s)) == c]


def main():
    frames = np.array([2, 3, 4, 5, 6, 7, 128])
    x0 = split_to_consecutive_v0(frames)

    print(x0)


if __name__ == '__main__':
    main()
