
import numpy as np
import xml.etree.ElementTree as ET
import os
import json
from torch.utils.data import Dataset
import yaml

def get_eaf(eaf_file: str):
    """
    Read an EAF file
    :param eaf_file: path to an EAF file
    :return: list of annotations (start_time, end_time, class)
    """

    tree = ET.parse(eaf_file)
    root = tree.getroot()

    # Get time in milliseconds
    time_slot_list = [time_slot.attrib for time_slot in root[1]]
    time_order = {slot['TIME_SLOT_ID']: int(slot['TIME_VALUE']) for slot in time_slot_list}

    # Get annotations
    annotation_list = [annotation for annotation in root[2]]
    annotations = []
    for annotation in annotation_list:
        time_slots = annotation[0].attrib['TIME_SLOT_REF1'], annotation[0].attrib['TIME_SLOT_REF2']
        value = annotation[0][0].text
        annotations.append((time_order[time_slots[0]], time_order[time_slots[1]], value))
    return annotations

class VideoDataset(Dataset):
    def __init__(self, meta_file, classes, load_from_json=None, frame_sample_rate=1, min_sequence_length=2,
                 max_sequence_length=16, input_fps=25, step=1000, num_threads=0, normalize=False):
        """
        Args:
            meta_file (`str`): Path to the metafile containing paths to video and annotation files
            classes (`list`): List of classes
            load_from_json (`str`, None): Path to the json file containing exported data by save_dataset_to_json
            frame_sample_rate (`float`): Frame sampling per second. (5 for processing frame each 5th second)
                                         If not result frame index not int, the number is floored (mainly for rate < 1)
            min_sequence_length (`int`): Minimum number of frames in a sequence
            max_sequence_length (`int`): Maximum number of frames in a sequence
            input_fps (`int`): Frame sampling of input video
            step (`int`): Number of annotation time steps in one second (1000 for milliseconds)
            num_threads (`int`): Number of threads for decord
            normalize (`bool`): Normalize the video
        """
        self.meta_file = meta_file
        self.classes = classes
        self.data = []
        self.frame_sample_rate = frame_sample_rate
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.input_fps = input_fps
        self.num_threads = num_threads
        self.step = step
        self.normalize = normalize
        sampling = self.input_fps * self.frame_sample_rate

        video_list = []
        self.meta_data = []
        with open(self.meta_file, 'r') as f:
            meta_data = json.load(f)
            for i, video in enumerate(meta_data):
                if video['video'] not in video_list:
                    video_list.append(video['video'])
                    self.meta_data.append(meta_data[i])

        if load_from_json is not None and os.path.exists(load_from_json):
            print(f'Loading data from {load_from_json}')
            with open(load_from_json, 'r') as f:
                self.data = json.load(f)
        else:
            for annotation_file in self.meta_data:
                annotation_list = get_eaf(annotation_file['annotation'])

                for annotation in annotation_list:
                    if annotation[2] not in self.classes:
                        continue

                    start_frame = int(annotation[0] / self.step * self.input_fps)
                    end_frame = int(annotation[1] / self.step * self.input_fps)
                    for i in range(int(np.ceil((end_frame - start_frame) /
                                               (self.max_sequence_length * self.input_fps * self.frame_sample_rate)))):
                        indexes = list(np.arange(start_frame + i * self.max_sequence_length * sampling,
                                                 start_frame + (i + 1) * self.max_sequence_length * sampling,
                                                 sampling).astype(int))
                        if indexes[-1] <= end_frame and len(indexes) >= self.min_sequence_length:
                            self.data.append([annotation_file['video'], indexes, self.classes.index(annotation[2])])
                        else:
                            last_len = (end_frame - start_frame) % (self.max_sequence_length * sampling)
                            indexes = list(np.arange(end_frame - last_len, end_frame, sampling).astype(int))
                            if len(indexes) >= self.min_sequence_length:
                                self.data.append([annotation_file['video'], indexes, self.classes.index(annotation[2])])

def get_label_on_idx(frame_idx, annotation_list):
    for annotation in annotation_list:
        if annotation[0] <= frame_idx <= annotation[1]:
            return annotation[2]
    else:
        return -1
    
class VideoStreamDataset(VideoDataset):
    def __init__(self, meta_file, classes, load_from_json=None, frame_sample_rate=1, context_size=8, overlap=2,
                 max_empty_frames=3, input_fps=25, step=1000, num_threads=0, normalize=False):
        """
        Args:
            meta_file (`str`): Path to the metafile containing paths to video and annotation files
            classes (`list`): List of classes
            load_from_json (`str`, None): Path to the json file containing exported data by save_dataset_to_json
            frame_sample_rate (`float`): Frame sampling per second. (5 for processing frame each 5th second)
                                         If not result frame index not int, the number is floored (mainly for rate < 1)
            context_size (`int`): Size of window (data) is 2 x context_size + 1 (center). 17 by default
            overlap (`int`): Overlap of sliding window while creating the dataset
            max_empty_frames (`int`): Maximum number of frames in a sequence that are not labeled with any class. -1 for unlimited
            input_fps (`int`): Frame sampling of input video
            step (`int`): Number of annotation time steps in one second (1000 for milliseconds)
            num_threads (`int`): Number of threads for decord
            normalize (`bool`): Normalize the video
        """

        self.meta_file = meta_file
        self.classes = classes
        self.data = []
        self.frame_sample_rate = frame_sample_rate
        self.context_size = context_size
        self.overlap = overlap
        self.max_empty_frames = max_empty_frames
        if self.max_empty_frames == -1:
            self.max_empty_frames = 9999
        self.input_fps = input_fps
        self.num_threads = num_threads
        self.step = step
        self.max_sequence_length = 2 * context_size + 1
        self.normalize = normalize
        sampling = self.input_fps * self.frame_sample_rate

        video_list = []
        self.meta_data = []
        with open(self.meta_file, 'r') as f:
            meta_data = json.load(f)
            for i, video in enumerate(meta_data):
                if video['video'] not in video_list:
                    video_list.append(video['video'])
                    self.meta_data.append(meta_data[i])

        if load_from_json is not None and os.path.exists(load_from_json):
            print(f'Loading data from {load_from_json}')
            with open(load_from_json, 'r') as f:
                self.data = json.load(f)
        else:
            for annotation_file in self.meta_data:
                annotation_list = get_eaf(annotation_file['annotation'])

                mid_frame = self.context_size * sampling

                while (mid_frame + self.context_size * sampling) / self.input_fps * self.step <= annotation_list[-1][1]:
                    indexes = list(np.arange(mid_frame - (self.context_size * sampling),
                                         mid_frame + ((self.context_size +1) * sampling),
                                         sampling).astype(int))
                    labels = [get_label_on_idx(i / self.input_fps * self.step, annotation_list) for i in indexes]
                    label = labels[self.context_size]
                    # label = get_label_on_idx(indexes[self.context_size] / self.input_fps * self.step, annotation_list)
                    if label == -1 or labels.count(-1) > self.max_empty_frames:
                        mid_frame += (self.context_size * 2 + 1 - self.overlap) * sampling
                        continue

                    if label not in self.classes:
                        mid_frame += (self.context_size * 2 + 1 - self.overlap) * sampling
                        continue

                    self.data.append([annotation_file['video'], indexes, self.classes.index(label)])
                    mid_frame += (self.context_size * 2 + 1 - self.overlap) * sampling
                    
                    
CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']
def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg
config = load_config('configs/config_local.yaml')
data_config = config['data']
                   
dataset = VideoStreamDataset(data_config['meta_file'], CLASSES)

from collections import Counter, defaultdict
import numpy as np

def analyze_stream_dataset(dataset, classes):
    total_class_counts = Counter()
    class_match_ratios = defaultdict(list)
    co_occurences = defaultdict(Counter)

    for video_name, indexes, label_idx in dataset.data:
        label = classes[label_idx]

        annotation_path = next(item for item in dataset.meta_data if item['video'] == video_name)['annotation']
        annotation_list = get_eaf(annotation_path)

        labels_in_window = []
        for frame_idx in indexes:
            time = frame_idx / dataset.input_fps * dataset.step
            lbl = get_label_on_idx(time, annotation_list)
            labels_in_window.append(lbl)

        # Vynech neanotované framy
        valid_labels = [lbl for lbl in labels_in_window if lbl != -1]
        if len(valid_labels) == 0:
            continue

        total_class_counts[label] += 1

        # Kolik framů má stejnou třídu jako středový label
        match_count = sum([1 for lbl in valid_labels if lbl == label])
        match_ratio = match_count / len(valid_labels)
        class_match_ratios[label].append(match_ratio)

        # Spočítej výskyty jiných tříd v rámci okna
        for other_label in valid_labels:
            if other_label != label:
                co_occurences[label][other_label] += 1

    print("\n Počty výskytů tříd (střed okna):")
    for cls in classes:
        print(f" - {cls}: {total_class_counts[cls]}")

    print("\n Průměrné pokrytí třídy ve validních framech okna:")
    for cls in classes:
        ratios = class_match_ratios[cls]
        if ratios:
            avg_ratio = np.mean(ratios)
            print(f" - {cls}: {avg_ratio:.2%} průměrné pokrytí (stejný label jako střed)")
        else:
            print(f" - {cls}: žádná data")

    print("\n Spolu-výskyty tříd v okně (když je střed '{label}'):")
    for cls in classes:
        co_counts = co_occurences[cls]
        total = sum(co_counts.values())
        if total > 0:
            print(f"\n  -> {cls}:")
            for other_cls in classes:
                if other_cls == cls:
                    continue
                count = co_counts[other_cls]
                if count > 0:
                    print(f"     {other_cls}: {count / total:.2%}")
            
analyze_stream_dataset(dataset, CLASSES)