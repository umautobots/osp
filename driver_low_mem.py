import numpy as np
import os
from utils.constants import DATASETS_ROOT, SAMPLE_DATASETS_ROOT
import utils.tt_dataset as tt
import evaluation.results as re
import evaluation.metrics as metrics


def main_driver():
    data_dir = os.path.join(SAMPLE_DATASETS_ROOT, 'tt_format/10hz/dut')
    n_obs = 30
    n_pred = 50
    dataset = tt.process_data2datasets(
        data_dir, n_obs, n_pred,
        dataset_id2file_id_list={0: range(0, 35)},
        valid_ids_kwargs=dict(exist_any_type=(tt.AgentType.ped,)),
    )
    metric_fcn_list = [
        ['E[d_t]', metrics.get_expected_dist_by_time_fcns(
           select_inds=np.arange(9, n_pred, 10))],
        ['rmse[t]', metrics.get_rmse_by_time_fcns(
            select_inds=np.arange(9, n_pred, 10))],
    ]
    is_display = False
    if is_display:
        import display.predictions as di
        import display.utils as du

    # setup for prediction methods
    from baselines import velocity_model
    from ss_model import predict
    from ss_model import predict_utils
    from misc import sf_predictions
    method_info = [
        (
           'CV',
           velocity_model.predict_constant_velocity,
           dict(n_steps=n_pred),
        ),
        (
            'SF',
            sf_predictions.predict,
            dict(n_steps=n_pred),
        ),
        (
           'OSP',
           predict.predict,
           dict(n_steps=n_pred, parameters=predict_utils.make_parameters_ind_train())
        ),
    ]
    prediction_methods = [re.TrajectoryResults(*info) for info in method_info]
    running_eval = re.RunningEvaluation(metric_fcn_list, prediction_methods)

    # predict
    print('{} sets in datasets'.format(len(dataset)))
    for i in range(len(dataset)):
        if (i % 100 == 0) and i > 0:
            print('\nCurrent metrics\n')
            running_eval.reduce()
            print('Predictions on {}'.format(i))
        ped_xy, vic_xy, dataset_id, datafile_id = dataset.get_df(i)
        ped_xy_obs, vic_xy_obs = ped_xy[:n_obs, ...], vic_xy[:n_obs, ...]

        for predict_fcn in prediction_methods:
            predict_fcn.predict(ped_xy_obs, vic_xy_obs, dataset_id, datafile_id)
        xy_true = ped_xy[n_obs:, ...]
        if is_display:
            di.display_predictions(
                ped_xy_obs, vic_xy_obs,
                xy_true,
                prediction_methods, prediction_ind=-1,
                data_title=du.format_example_title(
                    i, dataset.df_list[datafile_id].datafile_path),
            )
        running_eval.evaluate(prediction_methods, xy_true)
        # save mem
        for prediction_method in prediction_methods:
            prediction_method.clear()
    print('\n')
    running_eval.reduce(decimals=4)


if __name__ == '__main__':
    main_driver()
