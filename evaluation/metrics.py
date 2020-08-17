import numpy as np


def get_expected_dist_by_time_fcns(select_inds=np.arange(9, 60, 10)):
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        :param accumulator: list of average distances from previous examples
            list[i] = si, n_agents
        :param y_hats: n_steps, n_agents, 2, n_samples
        :param p: n_agents, n_samples | probabilities summing to 1
        :param y_true: n_steps, n_agents, 2
        :return: 
        """
        # si, n_agents, 2, n_samples
        difs = y_hats[select_inds, ...] - np.expand_dims(y_true[select_inds, ...], -1)
        dists = np.sqrt((difs ** 2).sum(axis=2))  # si, n_agents, n_samples
        expected_dist = np.einsum('ijk,jk->ij', dists, p)  # si, n_agents
        accumulator.append(expected_dist)

    def reduce_fcn(accumulator):
        expected_dists = np.concatenate(accumulator, axis=1)  # si, total agents
        return expected_dists.mean(axis=1)
    return init_fcn, accumulate_fcn, reduce_fcn


def get_rmse_by_time_fcns(select_inds=np.arange(9, 60, 10)):
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, **kwargs):
        """
        :param accumulator: list of squared distances from previous examples
            list[i] = si, n_agents
        :param y_hats: n_steps, n_agents, 2, n_samples
        :param p: n_agents, n_samples | probabilities summing to 1
        :param y_true: n_steps, n_agents, 2
        :return:
        """
        # si, n_agents, 2, n_samples
        difs = y_hats[select_inds, ...] - np.expand_dims(y_true[select_inds, ...], -1)
        dists = (difs ** 2).sum(axis=2)  # si, n_agents, n_samples
        expected_dist = np.einsum('ijk,jk->ij', dists, p)  # si, n_agents
        accumulator.append(expected_dist)

    def reduce_fcn(accumulator):
        expected_dists = np.concatenate(accumulator, axis=1)  # si, total agents
        return np.sqrt(expected_dists.mean(axis=1))
    return init_fcn, accumulate_fcn, reduce_fcn


def get_timing_fcns():
    init_fcn = lambda: []

    def accumulate_fcn(accumulator, y_hats, p, y_true, duration=np.nan, **kwargs):
        """
        :param accumulator: list of (duration, n_agents)
        :param y_hats: n_steps, n_agents, 2, n_samples
        :param p: n_agents, n_samples | probabilities summing to 1
        :param y_true: n_steps, n_agents, 2
        :param duration: time taken
        :return:
        """
        accumulator.append((duration, y_true.shape[1]))

    def reduce_fcn(accumulator):
        duration_n_agents = np.array(accumulator)
        return duration_n_agents[:, 0].mean()
    return init_fcn, accumulate_fcn, reduce_fcn
