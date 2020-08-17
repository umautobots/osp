import numpy as np


def single_prediction2probabilistic_format(y_hat, n_agent_dim=1):
    """
    Convert single-sample prediction to (dims, sample) array
    Also return single probability weighting
    :param y_hat: dims |
    :param n_agent_dim: index of n_agents in y_hat
    :return:
        y_hat: (*dims), n_samples
        p: n_agents, n_samples
    """
    return np.expand_dims(y_hat, -1),\
        np.ones((y_hat.shape[n_agent_dim], y_hat.shape[-1]))
