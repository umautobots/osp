import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def display_predictions(
        ped_xy_obs, vic_xy_obs, xy_true,
        prediction_methods, prediction_ind=-1, data_title=''):
    n = len(prediction_methods)
    fig, axes = plt.subplots(1, n, sharex='all', sharey='all', figsize=(12, 10))
    for i in range(n):
        display_single_method_prediction(
            axes[i], ped_xy_obs, vic_xy_obs, xy_true,
            prediction_methods[i], prediction_ind)
        axes[i].set(aspect='equal')
    fig.suptitle(data_title)
    plt.show()


def display_single_method_prediction(
        ax, ped_xy_obs, vic_xy_obs, xy_true,
        prediction_method, prediction_ind):
    # n_steps, n_agents, 2, n_samples
    xy_hat, p = prediction_method[prediction_ind][:2]
    fs = prediction_method.frame_skip
    n_ped = ped_xy_obs.shape[1]
    n_samples = xy_hat.shape[-1]
    for i in range(n_ped):
        xy_hat_alpha = 0.3 if n_samples == 1 else 0.1
        ax.plot(
            ped_xy_obs[:, i, 0], ped_xy_obs[:, i, 1],
            c='magenta', alpha=0.2
        )
        ax.plot(
            xy_hat[fs-1::fs, i, 0, :], xy_hat[fs-1::fs, i, 1, :],
            c='blue', alpha=xy_hat_alpha
        )
        ax.plot(
            xy_true[:, i, 0], xy_true[:, i, 1],
            c='black', ls='', marker='+', alpha=0.8
        )
    n_vic = vic_xy_obs.shape[1]
    for i in range(n_vic):
        ax.plot(
            vic_xy_obs[:, i, 0], vic_xy_obs[:, i, 1],
            c='orange', alpha=0.5
        )
        ax.plot(
            vic_xy_obs[-1, i, 0], vic_xy_obs[-1, i, 1],
            c='orange', marker='o', alpha=0.5
        )
    ax.set_title(prediction_method.name)


