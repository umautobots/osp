import numpy as np
from time import time
from .social_forces import predictions
from baselines import baseline_utils as ut


def predict(ped_xy, vic_xy, dataset_id, datafile_id, n_steps=50, **kwargs):
    # agents_xy: n_frames, n_agents, 2
    # predict only pedestrians
    n_peds = ped_xy.shape[1]
    abs_xy = np.concatenate((ped_xy, vic_xy), axis=1)
    t0 = time()
    # (n_preds, n_agents, 2)
    merged_preds = predictions.predict(abs_xy, n_predict=n_steps)
    td = time() - t0
    # print('elapsed {:.4f}'.format(time.time() - t0))
    merged_preds = merged_preds[:, :n_peds, :]
    return (*ut.single_prediction2probabilistic_format(merged_preds)), {'duration': td}





