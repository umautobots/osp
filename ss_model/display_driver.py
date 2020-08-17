import os
from utils.constants import DATASETS_ROOT
import utils.tt_dataset as tt
import display.data_explore as de
import display.utils as du
import matplotlib.pyplot as plt


def main_driver():
    data_dir = os.path.join(DATASETS_ROOT, 'raw/tt_format/10hz/dut')
    n_obs = 30
    n_pred = 50
    dataset = tt.process_data2datasets(
        data_dir, n_obs, n_pred,
        dataset_id2file_id_list={0: range(7, 12)}, is_fill_nan=True)

    print('{} sets in datasets'.format(len(dataset)))
    for i in range(len(dataset)):
        if not i % 20 == 0:
            continue
        ped_xy, vic_xy, dataset_id, datafile_id = dataset.get_df(i)
        ax = de.display_all(
            ped_xy, vic_xy,
            data_title=du.format_example_title(
                i, dataset.df_list[datafile_id].datafile_path)
        )
        de.display_v_partial(ax, ped_xy, vic_xy)
        de.display_r_is_not_set(ax, ped_xy, vic_xy)
        plt.show()


if __name__ == '__main__':
    main_driver()
