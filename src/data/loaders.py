import os
import glob
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

def read_aerdat(filepath):
    with open(filepath, mode='rb') as file:
        file_content = file.read()
    
    packet_format = 'BHHI'
    packet_size = struct.calcsize('=' + packet_format)
    num_events = len(file_content) // packet_size
    extra_bits = len(file_content) % packet_size

    if extra_bits:
        file_content = file_content[0:-extra_bits]

    event_list = list(struct.unpack('=' + packet_format * num_events, file_content))

    return event_list 

class EyeDataset:
    def __init__(self, data_dir, subject):
        self.data_dir = data_dir
        self.subject = subject

        self.frame_list = []
        self.event_list = []

    def __len__(self):
        return len(self.frame_list) + len(self.event_list)


    def __get_item__(self, index):
        frame_timestamp = self.frame_list[-1].timestamp
        event_timestamp = self.event_list[-4]

        if event_timestamp < frame_timestamp:
            polarity = self.event_list.pop()
            row = self.event_list.pop()
            col = self.event_list.pop()
            timestamp = self.event_list.pop()
            event = Event(polarity, row, col, timestamp)
            return event
        else:
            frame = self.frame_list.pop()
            img = Image.open(frame.img).convert("L")
            frame = frame._replace(img=img)
            return frame
    
    def collect_data(self, eye=0):
        print('Loading Frames...')
        self.frame_list = self.load_frame_data(eye)
        print('Number of frames: ' + str(len(self.frame_list)))
        print('Loading Events...')
        self.event_list = self.load_event_data(eye)
        print('Number of events: ' + str(len(self.event_list) // 4))
    
    def load_frame_data(self, eye):
        frame_list = []
        subject_name = "user" + str(self.subject)
        img_dir = os.path.join(self.data_dir, subject_name, str(eye), 'frames')
        img_filepaths = list(glob_imgs(img_dir))
        img_filepaths.sort(key=lambda path: get_path_info(path)['index'])
        
        for fpath in img_filepaths:
            path_info = get_path_info(fpath)
            frame = Frame(path_info['row'], path_info['col'], fpath, path_info['timestamp'])
            frame_list.append(frame)
        return frame_list

    def load_event_data(self, eye):
        subject_name = "user" + str(self.subject)
        event_file = os.path.join(self.data_dir, subject_name, str(eye), 'events.aerdat')
        events_list = read_aerdat(event_file)
        return events_list