import numpy as np
from baselines import baseline_utils as ut
from time import time


def predict_constant_velocity(
        ped_xy, vic_xy, dataset_id, datafile_id,
        n_steps=0, **kwargs):
    """
    Predict for the pedestrians only
    :param ped_xy: n_frames, n_agents, 2
    :param vic_xy: 
    :param dataset_id: 
    :param datafile_id: 
    :param n_steps: 
    :param kwargs: 
    :return:
        xy_hat: n_steps, n_agents, 2, n_samples=1
        p: n_agents, n_samples
        data_dict:
    """
    t0 = time()
    vdt = ped_xy[1:] - ped_xy[:-1]
    mean_vdt = vdt.mean(axis=0)  # n_agents, 2
    xy_hat = ped_xy[[-1]] + mean_vdt[np.newaxis, :, :] * np.arange(1, n_steps+1)[:, np.newaxis, np.newaxis]
    return (*ut.single_prediction2probabilistic_format(xy_hat)), {'duration': time()-t0}

