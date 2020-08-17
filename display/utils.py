import os


def format_example_title(i, datafile_path):
    name = datafile_path2name(datafile_path)
    title_str = '{}\nset {}'.format(name, i)
    return title_str


def format_agent_title(agent_id, datafile_path):
    name = datafile_path2name(datafile_path)
    title_str = '{}\nagent {}'.format(name, agent_id)
    return title_str


def format_frame_window_title(frame_0, n_frames, datafile_path):
    name = datafile_path2name(datafile_path)
    title_str = '{}\nframes {}:{}'.format(
        name, frame_0, frame_0 + n_frames)
    return title_str


def datafile_path2name(datafile_path):
    name = os.path.basename(datafile_path)
    name = name.replace('.csv', '').replace('.txt', '')
    return name
