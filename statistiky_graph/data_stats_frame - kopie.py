import xml.etree.ElementTree as ET
import json
import math
import cv2
import numpy as np
CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']


def get_eaf(eaf_file: str):
    """
    Read an EAF file and extract all annotations.
    :param eaf_file: path to an EAF file
    :return: list of annotations (start_time, end_time, class)
    """
    tree = ET.parse(eaf_file)
    root = tree.getroot()

    # Extract TIME_ORDER
    time_order_element = root.find("TIME_ORDER")
    time_order = {
        time_slot.attrib['TIME_SLOT_ID']: int(time_slot.attrib['TIME_VALUE'])
        for time_slot in time_order_element.findall("TIME_SLOT")
    }

    annotations = []

    # Extract all TIERs
    for tier in root.findall("TIER"):
        for annotation in tier.findall("ANNOTATION"):
            alignable = annotation.find("ALIGNABLE_ANNOTATION")
            if alignable is not None:
                start_ts = alignable.attrib["TIME_SLOT_REF1"]
                end_ts = alignable.attrib["TIME_SLOT_REF2"]
                value = alignable.find("ANNOTATION_VALUE").text
                annotations.append((time_order[start_ts], time_order[end_ts], value))

    return annotations

def get_video_duration_ms(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps == 0:
        raise ValueError(f"FPS is zero for video: {video_path}")
    
    duration_ms = (frame_count / fps) * 1000
    return duration_ms

def analyze_all_videos(meta_file, fps=25):
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)

    total_frames = 0
    total_annotated_frames = 0
    frames_per_class = {cls: 0 for cls in CLASSES}
    class_instances = {cls: 0 for cls in CLASSES}
    class_instance_lengths = {cls: [] for cls in CLASSES}
    unannotated_frames = 0

    for entry in meta_data:
        annotation_file = entry['annotation']
        video_path = entry['video']

        try:
            video_duration = get_video_duration_ms(video_path)
        except Exception as e:
            print(f"Chyba při čtení videa {video_path}: {e}")
            continue

        annotations = get_eaf(annotation_file)
        num_frames = int(np.ceil(video_duration / 1000 * fps))
        frame_labels = ['none'] * num_frames

        for start_ms, end_ms, cls in annotations:
            if cls not in CLASSES:
                continue
            start_frame = int(start_ms / 1000 * fps)
            end_frame = int(end_ms / 1000 * fps)
            for i in range(start_frame, min(end_frame, num_frames)):
                frame_labels[i] = cls

        total_frames += num_frames
        annotated_frames = sum(1 for lbl in frame_labels if lbl != 'none')
        total_annotated_frames += annotated_frames
        unannotated_frames += num_frames - annotated_frames

        for cls in CLASSES:
            frames_per_class[cls] += sum(1 for lbl in frame_labels if lbl == cls)

        prev_cls = 'none'
        length = 0
        for lbl in frame_labels:
            if lbl != prev_cls:
                if prev_cls in CLASSES and length > 0:
                    class_instances[prev_cls] += 1
                    class_instance_lengths[prev_cls].append(length)
                length = 1
                prev_cls = lbl
            else:
                length += 1
        if prev_cls in CLASSES and length > 0:
            class_instances[prev_cls] += 1
            class_instance_lengths[prev_cls].append(length)

    print("Celkový počet framů:", total_frames)
    print("Počet anotovaných framů:", total_annotated_frames)
    print("Počet neanotovaných framů:", unannotated_frames)
    print("\nPočet framů na třídu:")
    for cls in CLASSES:
        print(f"  {cls}: {frames_per_class[cls]}")
    print("\nPočet instancí na třídu:")
    for cls in CLASSES:
        print(f"  {cls}: {class_instances[cls]}")
    print("\nStatistiky délek instancí:")
    for cls in CLASSES:
        lengths = class_instance_lengths[cls]
        if lengths:
            print(f"  {cls}: avg = {np.mean(lengths):.2f}, min = {min(lengths)}, max = {max(lengths)}")
        else:
            print(f"  {cls}: žádné instance")


analyze_all_videos("RAVDAI/metadata/metadata_local.json", fps=25)