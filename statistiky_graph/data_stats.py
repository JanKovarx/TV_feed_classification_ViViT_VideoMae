import os
import cv2
from collections import defaultdict
import torch
from dataset import VideoDataset
import yaml
import json
from collections import defaultdict, Counter

CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']

# Cesta ke složce s videi
VIDEO_FOLDER = "RAVDAI\\data-240p"

def load_config(cfg_path):
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

def analyze_videos(dataset):
    total_videos = len(dataset)
    total_length = 0
    class_counts = defaultdict(int)
    transition_counts = defaultdict(lambda: defaultdict(int))
    video_class_sets = []  # Seznam množin tříd pro každé video

    print("Analýza videí:")

    for idx, video in enumerate(dataset):
        frames, labels, _ = video
        total_length += len(frames)

        unique_classes = set()
        previous_class = None

        for label in labels:
            class_counts[label] += 1
            unique_classes.add(label)

            # Přechod mezi třídami
            if previous_class is not None:
                transition_counts[previous_class][label] += 1
            previous_class = label

        video_class_sets.append(tuple(sorted(unique_classes)))

    # Shrnutí
    average_video_length = total_length / total_videos if total_videos > 0 else 0
    total_samples = sum(class_counts.values())
    unique_total_classes = set(class_counts.keys())

    print(f"Počet videí: {total_videos}")
    print(f"Průměrná délka videí: {average_video_length:.2f} rámců")
    print(f"Počet tříd v celém datasetu: {len(unique_total_classes)}")

    print("\nPočet přechodů mezi třídami:")
    for start_class in transition_counts:
        for end_class in transition_counts[start_class]:
            count = transition_counts[start_class][end_class]
            if count > 0:
                print(f"Transition from {CLASSES[start_class]} to {CLASSES[end_class]}: {count}×")

    # Spočítej frekvenci výskytů konkrétních kombinací tříd ve videích
    combination_counter = Counter(video_class_sets)
    combination_summary = {
        ' + '.join([CLASSES[c] for c in combo]): count
        for combo, count in combination_counter.items()
    }

    # --- Uložení do JSON ---
    results = {
        "total_videos": total_videos,
        "average_video_length": average_video_length,
        "total_annotated_frames": total_samples,
        "unique_total_classes_count": len(unique_total_classes),
        "class_counts": {CLASSES[label]: count for label, count in class_counts.items()},
        "transition_counts": {
            CLASSES[start]: {
                CLASSES[end]: count for end, count in end_dict.items()
            }
            for start, end_dict in transition_counts.items()
        },
        "video_class_combinations": combination_summary
    }

    with open("vysledky_analyzy_videa.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        


def analyze_classes(dataset):
    class_instance_lengths = defaultdict(list)  # Délka jednotlivých instancí tříd
    class_instance_counts = defaultdict(int)    # Počet instancí tříd
    unannotated_instances = 0

    print("\nAnalýza tříd:")
    
    for video in dataset:
        frames, labels, _ = video
        previous_class = None
        instance_start = None

        # Procházení jednotlivých rámců a tříd
        for i, (frame, label) in enumerate(zip(frames, labels)):
            # Pokud je neanotovaná třída, zvyšujeme počet neanotovaných instancí
            if label == -1:
                unannotated_instances += 1
                continue

            class_instance_counts[label] += 1
            
            # Počítání délky instancí tříd
            if previous_class != label:
                if previous_class is not None and instance_start is not None:
                    class_instance_lengths[previous_class].append(i - instance_start)
                instance_start = i  # Nová instance třídy
            previous_class = label

        # Na konci videa přidej poslední instanci
        if previous_class is not None and instance_start is not None:
            class_instance_lengths[previous_class].append(len(frames) - instance_start)

    # Počítání průměrné délky instancí pro každou třídu
    for class_label in class_instance_lengths:
        lengths = class_instance_lengths[class_label]
        average_length = sum(lengths) / len(lengths) if lengths else 0
        print(f"Průměrná délka instancí třídy {CLASSES[class_label]}: {average_length:.2f} rámců")

    print(f"Počet neanotovaných instancí: {unannotated_instances}")
    
    return class_instance_counts, class_instance_lengths, unannotated_instances

def video_length_analysis():
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
    num_videos = len(video_files)
    video_lengths = {}
    stanice_info = defaultdict(lambda: {'count': 0, 'total_duration': 0})

    for video in video_files:
        video_path = os.path.join(VIDEO_FOLDER, video)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        duration_minutes = duration / 60
        video_lengths[video] = duration_minutes

        stanice = video.split()[0]  # Rozdělení podle mezery
        stanice_info[stanice]['count'] += 1
        stanice_info[stanice]['total_duration'] += duration_minutes

    celkova_delka = sum([length for length in video_lengths.values()])
    celkova_delka_hodiny = celkova_delka / 60

    print(f"\nPočet videí: {num_videos}")
    for video, length in video_lengths.items():
        print(f"{video}: {length:.2f} minut")

    print("\nStatistiky pro jednotlivé stanice:")
    for stanice, info in stanice_info.items():
        print(f"{stanice}: {info['count']} záznamů, Celková délka: {info['total_duration']:.2f} minut")

    print(f"\nCelková délka všech videí: {celkova_delka_hodiny:.3f} hodin")
    return num_videos, video_lengths, stanice_info

def dataset_distribution(dataset):
    class_counts, transitions, unannotated_frames, total_samples = analyze_videos(dataset)
    class_instance_counts, class_instance_lengths, unannotated_instances = analyze_classes(dataset)
    
    print("\nPočet instancí tříd:")
    for class_label, count in class_instance_counts.items():
        print(f"Třída {CLASSES[class_label]}: {count} instancí")
    
    return {
        "class_counts": class_counts,
        "transitions": transitions,
        "unannotated_frames": unannotated_frames,
        "total_samples": total_samples,
        "class_instance_counts": class_instance_counts,
        "class_instance_lengths": class_instance_lengths,
        "unannotated_instances": unannotated_instances
    }

# Hlavní část programu
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())

# Načtení konfigurace
config = load_config('configs/config_local.yaml')
data_config = config['data']

# Vytvoření datasetu
print('Loading dataset...')
dataset = VideoDataset(data_config['meta_file'], CLASSES, frame_sample_rate=data_config['frame_sample_rate'],
                        min_sequence_length=data_config['min_sequence_length'],
                        max_sequence_length=data_config['max_sequence_length'],
                        video_decoder=data_config['video_decoder'],)

print('Loading done')

# Analýza videí a tříd
dataset_info = dataset_distribution(dataset)

# Analýza videí ze složky
video_length_info = video_length_analysis()