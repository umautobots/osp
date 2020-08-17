import utils.tt_dataset as tt
from utils.constants import DATASETS_ROOT
import os
from glob import glob
import pandas as pd


class IndDataset(object):
    """
    *tracksMeta.csv
    recordingId 	The id of the recording. Every recording has a unique id. 	[-]
    trackId 	The id of the track. The ids are assigned in ascending order for each recording. 	[-]
    initialFrame 	The frame in which the track starts. 	[-]
    finalFrame 	The frame in which the track ends. 	[-]
    numFrames 	The total lifetime in frames. 	[-]
    width 	The width of the tracked object. 0.0 for VRUs. 	[m]
    length 	The length of the tracked object. 0.0 for VRUs. 	[m]
    class
    -class='car', 'truck_bus', 'pedestrian', 'bicycle'
    
    *tracks.csv
    recordingId,trackId,frame,
    trackLifetime,xCenter,yCenter,
    heading,width,length,xVelocity,yVelocity,
    xAcceleration,yAcceleration,
    lonVelocity,latVelocity,lonAcceleration,latAcceleration
    - 25Hz
    """
    FOLDER = 'raw/ind19/data'
    TYPE_NAME2TT_TYPE_ID = {
        'pedestrian': tt.AgentType.ped,
        'car': tt.AgentType.vic,
        'truck_bus': tt.AgentType.vic,
        'bicycle': tt.AgentType.bicycle,
    }
    RAW_DT = 1./25
    GLOB_STR = '*_tracks.csv'

    @staticmethod
    def load_raw(track_path):
        track_df = pd.read_csv(track_path, sep=',')
        track_df = track_df.astype({'trackId': int, 'frame': int})
        meta_path = track_path.replace('tracks.', 'tracksMeta.')
        meta_df = pd.read_csv(meta_path, sep=',')
        meta_df = meta_df.astype({'trackId': int})
        return track_df, meta_df

    @staticmethod
    def raw2tt(track_df, meta_df, offset_agent_id=0):
        df = track_df.merge(meta_df[['trackId', 'class']], on='trackId', how='left')
        tt_df = df.rename(columns={
            'frame': 'frame_id',
            'trackId': 'agent_id',
            'xCenter': 'x',
            'yCenter': 'y',
        }, inplace=False)
        tt_df['type_id'] = -1
        for label_key, type_id in IndDataset.TYPE_NAME2TT_TYPE_ID.items():
            tt_df.loc[tt_df['class'].str.match(label_key), 'type_id'] = type_id
        tt_df['agent_id'] += offset_agent_id
        tt_df = tt_df[['frame_id', 'agent_id', 'x', 'y', 'type_id']]
        tt.format_dataframe(tt_df, is_raise=False)
        return tt_df

    def get_recordings(self):
        glob_str = os.path.join(
            DATASETS_ROOT, self.FOLDER,
            self.GLOB_STR
        )
        for track_path in glob(glob_str):
            name = os.path.basename(track_path)
            name = name.replace('_tracks.csv', '')
            yield  name, track_path

    def load_as_trajectorytype_format(self, track_path):
        track_df, meta_df = self.load_raw(track_path)
        df = self.raw2tt(track_df, meta_df)
        return df

