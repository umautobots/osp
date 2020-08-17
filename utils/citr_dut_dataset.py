import utils.tt_dataset as tt
from utils.constants import DATASETS_ROOT
import os
from glob import glob
import pandas as pd


class CitrDutDataset(object):
    """
    - separate files - ped vs vehicle
    - not always a vehicle file, but always a pedestrian file
    - *ped_filtered.csv
    id,frame,label,x_est,y_est,vx_est,vy_est
    - veh_filtered.csv
    id,frame,label,x_est,y_est,psi_est,vel_est
    - label='ped'/'veh'
    CITR = 29.97Hz
    DUT = 23.98Hz
    """
    FOLDER = ''
    GLOB_STR = None
    TYPE_NAME2TT_TYPE_ID = {
        'ped': tt.AgentType.ped,
        'veh': tt.AgentType.vic,
    }

    @staticmethod
    def load_raw(p):
        df = pd.read_csv(p, sep=',')
        df = df.astype({'id': int, 'frame': int})
        return df

    @staticmethod
    def raw2tt(df, offset_agent_id=0):
        tt_df = df.rename(columns={
            'frame': 'frame_id',
            'id': 'agent_id',
            'x_est': 'x',
            'y_est': 'y',
        }, inplace=False)
        tt_df['type_id'] = -1
        for label_key, type_id in CitrDutDataset.TYPE_NAME2TT_TYPE_ID.items():
            tt_df.loc[tt_df['label'].str.match(label_key), 'type_id'] = type_id
        tt_df['agent_id'] += offset_agent_id
        tt_df = tt_df[['frame_id', 'agent_id', 'x', 'y', 'type_id']]
        tt.format_dataframe(tt_df, is_raise=False)
        return tt_df

    def get_recordings(self):
        glob_str = os.path.join(
            DATASETS_ROOT, self.FOLDER,
            self.GLOB_STR
        )
        for ped_path in glob(glob_str):
            name = os.path.basename(ped_path)
            name = name.replace('_traj_ped_filtered.csv', '')
            vic_path = ped_path.replace('ped', 'veh')
            if not os.path.exists(vic_path):
                vic_path = None
            yield  name, (ped_path, vic_path)

    def load_as_trajectorytype_format(self, recording):
        ped_path, vic_path = recording
        df = self.load_raw(ped_path)
        df = self.raw2tt(df)
        if vic_path:
            agent_id_offset = df['agent_id'].max() + 1
            vic_df = self.load_raw(vic_path)
            df = df.append(self.raw2tt(vic_df, agent_id_offset))
        return df


class CitrDataset(CitrDutDataset):
    FOLDER = 'raw/vci-dataset-citr/data/trajectories_filtered'
    RAW_DT = 1./29.97
    GLOB_STR = '*/*_ped_filtered.csv'


class DutDataset(CitrDutDataset):
    FOLDER = 'raw/vci-dataset-dut/data/trajectories_filtered'
    RAW_DT = 1./23.98
    GLOB_STR = '*_ped_filtered.csv'



