import os
import glob
import json
import struct

import numpy as np

from collections import namedtuple
from PIL import Image

'Types of data'
Event = namedtuple('Event', 'polarity row col timestamp')
Frame = namedtuple('Frame', 'row col img timestamp')

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob.glob(os.path.join(path,'**', ext), recursive=True))
    return imgs

def get_path_info(path):
    path_parts = path.split('/')[-1].split('.')[0].split('_')
    return {'index': int(path_parts[0]), 'row': int(path_parts[1]), 'col': int(path_parts[2]),
            'stimulus_type': path_parts[3], 'timestamp': int(path_parts[4])}

def read_aerdat(filepath, mode):
    with open(filepath, mode='rb') as file:
        file_content = file.read()

    packet_format = 'BHHI'
    packet_size = struct.calcsize('=' + packet_format)
    num_events = len(file_content) // packet_size
    extra_bits = len(file_content) % packet_size

    if extra_bits:
        file_content = file_content[0:-extra_bits]

    event_list = list(struct.unpack('=' + packet_format * num_events, file_content))
    if mode == 'stack':
        event_list.reverse()
        return event_list
    elif mode == 'np':
        events = np.array(event_list, dtype=np.uint32).reshape(-1, 4)
        return events
    else:
        return event_list

def read_events_txt(filepath, mode):
    """Parse ev_eye text events: 'timestamp row col polarity' per line.
    Returns same layout as read_aerdat: columns [polarity, row, col, timestamp].
    """
    data = np.loadtxt(filepath, dtype=np.int64)  # (N, 4): ts, row, col, pol
    events = data[:, [3, 1, 2, 0]]               # reorder to: pol, row, col, ts
    if mode == 'np':
        return events
    elif mode == 'stack':
        flat = events[::-1].flatten().tolist()
        return flat
    return events

class EyeDataset:
    def __init__(self, data_dir, subject, mode='stack'):
        self.data_dir = data_dir
        self.subject = subject
        self.mode = mode

        self.frame_list = []

        self.event_list = None  # Used by np mode
        self.event_stack = []   # Used by stack mode

    def __len__(self):
        return len(self.frame_list) + len(self.event_list)


    def __get_item__(self, _):
        frame_timestamp = self.frame_list[-1].timestamp
        event_timestamp = self.event_stack[-4]

        if event_timestamp < frame_timestamp:
            polarity = self.event_stack.pop()
            row = self.event_stack.pop()
            col = self.event_stack.pop()
            timestamp = self.event_stack.pop()
            event = Event(polarity, row, col, timestamp)
            return event
        else:
            frame = self.frame_list.pop()
            # img = Image.open(frame.img).convert("L")
            # frame = frame._replace(img=img)
            return frame

    def get_initial_frame(self):
        while(1):
            itm = self.__get_item__(0)
            if type(itm) is Frame:
                return itm

    def load_events_sorted(self, eye=0):
        """Return events as numpy array (N, 4) = [polarity, row, col, timestamp], sorted by timestamp."""
        subject_name = "user" + str(self.subject)
        event_file = os.path.join(self.data_dir, subject_name, str(eye), 'events.aerdat')
        events = read_aerdat(event_file, mode='np')  # shape (N, 4)
        return events[events[:, 3].argsort()]

    def collect_data(self, eye=0, motion='saccadic'):
        print('Loading Frames...')
        self.frame_list = self.load_frame_data(eye, motion)
        print('Number of frames: ' + str(len(self.frame_list)))
        print('Loading Events...')
        events = self.load_event_data(eye)
        if self.mode == 'stack':
            self.event_stack = events
            print('Number of events: ' + str(int(len(self.event_stack) / 4)))
        elif self.mode == 'np':
            self.event_list = events
            print('Number of events: ' + str(len(self.event_list)))

    def load_frame_data(self, eye, motion='saccadic'):
        stimulus_type = 's' if motion == 'saccadic' else 'p'
        frame_list = []
        subject_name = "user" + str(self.subject)
        img_dir = os.path.join(self.data_dir, subject_name, str(eye), 'frames')
        img_filepaths = list(glob_imgs(img_dir))
        img_filepaths = [p for p in img_filepaths
                         if get_path_info(p)['stimulus_type'] == stimulus_type]
        img_filepaths.sort(key=lambda path: get_path_info(path)['index'])

        if self.mode == 'stack':
            img_filepaths.reverse()

        for fpath in img_filepaths:
            path_info = get_path_info(fpath)
            frame = Frame(path_info['row'], path_info['col'], fpath, path_info['timestamp'])
            frame_list.append(frame)
        return frame_list

    def load_event_data(self, eye):
        subject_name = "user" + str(self.subject)
        event_file = os.path.join(self.data_dir, subject_name, str(eye), 'events.aerdat')
        events_list = read_aerdat(event_file, self.mode)
        return events_list


class EvEyeDataset:
    SACCADIC_SESSIONS = ['session_1_0_1', 'session_1_0_2']
    PURSUIT_SESSIONS  = ['session_2_0_1', 'session_2_0_2']

    def __init__(self, data_dir, subject, motion='saccadic', mode='np',
                 screen_width_px=1920, screen_height_px=1080):
        self.data_dir = data_dir
        self.subject = subject
        self.motion = motion
        self.mode = mode
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.sessions = (self.SACCADIC_SESSIONS if motion == 'saccadic'
                         else self.PURSUIT_SESSIONS)

        self.frame_list = []
        self.event_list = None
        self.event_stack = []

    def _subject_name(self):
        return f'user{self.subject}'

    def _davis_session_dir(self, eye, session):
        return os.path.join(self.data_dir, 'data_davis', self._subject_name(), eye, session)

    def _tobii_session_dir(self, session):
        return os.path.join(self.data_dir, 'data_tobii', self._subject_name(), session)

    def _load_gaze_records_for_session(self, eye, session):
        """Return sorted array (M, 3): [davis_us, x_norm, y_norm] for one session."""
        startime_path = os.path.join(
            self._davis_session_dir(eye, session), 'events', 'event_startime.txt')
        with open(startime_path) as f:
            startime_us = int(f.read().strip())

        gazedata_path = os.path.join(self._tobii_session_dir(session), 'gazedata')
        records = []
        with open(gazedata_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    gaze2d = entry.get('data', {}).get('gaze2d')
                    if gaze2d is not None:
                        davis_us = startime_us + int(entry['timestamp'] * 1e6)
                        records.append([davis_us, gaze2d[0], gaze2d[1]])
                except (json.JSONDecodeError, KeyError):
                    continue
        if not records:
            return np.empty((0, 3))
        arr = np.array(records, dtype=np.float64)
        return arr[arr[:, 0].argsort()]

    def _load_gaze_labels(self, eye):
        """Return sorted array (N, 3): [davis_us, x_norm, y_norm] across all sessions."""
        parts = [self._load_gaze_records_for_session(eye, s) for s in self.sessions]
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            return np.empty((0, 3))
        combined = np.vstack(parts)
        return combined[combined[:, 0].argsort()]

    def load_frame_data(self, eye):
        """Return list of (timestamp, path) sorted chronologically across all sessions."""
        all_frames = []
        for session in self.sessions:
            frames_dir = os.path.join(self._davis_session_dir(eye, session), 'frames')
            for path in glob_imgs(frames_dir):
                fname = os.path.splitext(os.path.basename(path))[0]
                parts = fname.split('_')
                ts = int(parts[1])
                all_frames.append((ts, path))
        all_frames.sort(key=lambda x: x[0])
        return all_frames

    def load_event_data(self, eye):
        """Return events (N, 4) = [polarity, row, col, timestamp] sorted by timestamp."""
        parts = []
        for session in self.sessions:
            events_path = os.path.join(
                self._davis_session_dir(eye, session), 'events', 'events.txt')
            parts.append(read_events_txt(events_path, mode='np'))
        combined = np.vstack(parts)
        return combined[combined[:, 3].argsort()]

    def load_events_sorted(self, eye=None):
        """Return events sorted by timestamp. Re-uses cached event_list if available."""
        if self.event_list is not None:
            return self.event_list
        return self.load_event_data(eye or 'left')

    def collect_data(self, eye='left', motion=None):
        print('Loading Frames...')
        raw_frames = self.load_frame_data(eye)  # [(ts, path), ...]
        print(f'Number of frames: {len(raw_frames)}')

        print('Loading Gaze Labels...')
        gaze_records = self._load_gaze_labels(eye)  # (M, 3): davis_us, x_norm, y_norm
        if len(gaze_records) == 0:
            raise RuntimeError(
                f"No Tobii gaze records found for subject {self.subject}, eye {eye}")

        frame_ts = np.array([ts for ts, _ in raw_frames], dtype=np.int64)
        gaze_ts  = gaze_records[:, 0].astype(np.int64)

        idx      = np.clip(np.searchsorted(gaze_ts, frame_ts), 0, len(gaze_records) - 1)
        prev_idx = np.maximum(idx - 1, 0)
        best     = np.where(np.abs(gaze_ts[prev_idx] - frame_ts) <
                            np.abs(gaze_ts[idx]      - frame_ts), prev_idx, idx)

        frame_list = []
        for i, (ts, path) in enumerate(raw_frames):
            gi  = best[i]
            col = int(round(gaze_records[gi, 1] * self.screen_width_px))
            row = int(round(gaze_records[gi, 2] * self.screen_height_px))
            frame_list.append(Frame(row, col, path, ts))

        # Storage order: newest first (reverse of chronological)
        self.frame_list = frame_list[::-1]

        print('Loading Events...')
        self.event_list = self.load_event_data(eye)
        print(f'Number of events: {len(self.event_list)}')
