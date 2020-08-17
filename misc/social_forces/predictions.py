"""
https://github.com/vita-epfl/trajnetplusplusbaselines
"""
import numpy as np
from scipy.interpolate import interp1d

from .simulator import Simulator
from .potentials import PedPedPotential
from .fieldofview import FieldOfView


def predict(input_paths, dest_dict=None, dest_type='interp', sf_params=(0.5, 2.1, 0.3),
            predict_all=True, n_predict=50, obs_length=30):
    """
    Predict given data as input_paths, conditioning on data in frames before obs_length
    :param input_paths:
    formerly:
        n_agents, t | with attributes:
        x:
        y:
        frame:
        pedestrian: id
    now:
        n_frames, n_agents, 2 |
        xy: [..., :2]
        frame: equal to t, [i, ...]
        id: equal to index [:, i, :]
    :param dest_dict:
    :param dest_type:
    :param sf_params:
    :param predict_all:
    :param n_predict:
    :param obs_length:
    :return:
        n_predict, n_agents, 2
    """
    pred_length = n_predict
    fps = 20  # simulation fps
    input_fps = 10
    desired_fps = 10
    sampling_rate = int(fps / desired_fps)

    def init_states(input_paths, start_frame, dest_dict, dest_type):
        n_frames, n_agents = input_paths.shape[:2]
        initial_state = []
        for i in range(n_agents):
            path = input_paths[:, i, :]  # n_frames, 2
            ped_id = i
            past_path = path[:start_frame, :]
            future_path = path[start_frame:, :]
            len_path = past_path.shape[0]

            ## To consider agent or not consider.
            if start_frame < n_frames:
                curr = past_path[-1]

                ## Velocity
                if len_path >= 4:
                    stride = 3
                    prev = past_path[-4]
                else:
                    stride = len_path - 1
                    prev = past_path[-len_path]
                [v_x, v_y] = vel_state(prev, curr, stride)

                ## Destination
                if dest_type == 'true':
                    if dest_dict is not None:
                        [d_x, d_y] = dest_dict[ped_id]
                    else:
                        raise ValueError
                elif dest_type == 'interp':
                    [d_x, d_y] = dest_state(past_path, len_path)
                elif dest_type == 'vel':
                    [d_x, d_y] = [pred_length * v_x, pred_length * v_y]
                elif dest_type == 'pred_end':
                    [d_x, d_y] = [future_path[-1, 0], future_path[-1, 1]]
                else:
                    raise NotImplementedError

                ## Initialize State
                initial_state.append([curr[0], curr[1], v_x, v_y, d_x, d_y])
        return np.array(initial_state)

    def vel_state(prev, curr, stride):
        """
        :param prev: 2, | xy
        :param curr: 2,
        :param stride: number of frames prev, curr are apart
        :return:
        """
        if stride == 0:
            return [0, 0]
        diff = np.array([curr[0] - prev[0], curr[1] - prev[1]])
        theta = np.arctan2(diff[1], diff[0])
        speed = np.linalg.norm(diff) / (stride * 1 / input_fps)
        return [speed * np.cos(theta), speed * np.sin(theta)]

    def dest_state(xy, length):
        if length == 1:
            return [xy[-1, 0], xy[-1, 1]]
        f = interp1d(x=np.arange(length), y=xy.T, fill_value='extrapolate')
        return f(length-1 + pred_length)

    # initialize
    start_frame = obs_length - 1
    initial_state = init_states(input_paths, start_frame, dest_dict, dest_type)
    if len(initial_state) != 0:
        # run
        ped_ped = PedPedPotential(1. / fps, v0=sf_params[1], sigma=sf_params[2])
        field_of_view = FieldOfView()
        s = Simulator(initial_state, ped_ped=ped_ped, field_of_view=field_of_view,
                      delta_t=1. / fps, tau=sf_params[0])
        states = np.stack([s.step().state.copy() for _ in range(pred_length * sampling_rate)])
        ## states : pred_length x num_ped x 7
        states = np.array([s for num, s in enumerate(states) if num % sampling_rate == 0])
    else:
        ## Stationary
        # note: appears to be only for single agent, raised if start_frame >= n_frames
        past_path = input_paths[:start_frame, 0, :]
        states = np.stack([[[past_path[0, 0], past_path[0, 1]]] for _ in range(pred_length)])

    predictions = states[..., :2]
    return predictions
