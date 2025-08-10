import os
import cv2
from collections import defaultdict
import json

# Cesty
VIDEO_FOLDER = "RAVDAI\\data-240p"
METADATA_FILE = "RAVDAI\\metadata\\metadata_local.json"

def video_length_analysis():
    # Načti metadata a extrahuj názvy videí
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    valid_videos = set(os.path.basename(item["video"]) for item in metadata)

    # Vyfiltruj pouze videa uvedená v metadatech
    video_files = [f for f in os.listdir(VIDEO_FOLDER)
                   if f.endswith((".mp4", ".avi", ".mov", ".mkv")) and f in valid_videos]

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

        stanice = video.split()[0]
        stanice_info[stanice]['count'] += 1
        stanice_info[stanice]['total_duration'] += duration_minutes

    celkova_delka = sum(video_lengths.values())
    celkova_delka_hodiny = celkova_delka / 60
    prumerna_delka = celkova_delka / num_videos if num_videos > 0 else 0

    print(f"\nPočet videí v metadatech: {num_videos}")
    for video, length in video_lengths.items():
        print(f"{video}: {length:.2f} minut")

    print("\nStatistiky pro jednotlivé stanice:")
    for stanice, info in stanice_info.items():
        print(f"{stanice}: {info['count']} záznamů, Celková délka: {info['total_duration']:.2f} minut")

    print(f"\nCelková délka všech videí: {celkova_delka_hodiny:.3f} hodin")
    print(f"Průměrná délka videa: {prumerna_delka:.2f} minut")

    # Uložení do JSON
    output = {
        "pocet_videi": num_videos,
        "celkova_delka_minuty": round(celkova_delka, 2),
        "celkova_delka_hodiny": round(celkova_delka_hodiny, 3),
        "prumerna_delka_minuty": round(prumerna_delka, 2),
        "video_lengths": {video: round(length, 2) for video, length in video_lengths.items()},
        "stanice_info": {
            stanice: {
                "count": info["count"],
                "total_duration": round(info["total_duration"], 2)
            }
            for stanice, info in stanice_info.items()
        }
    }

    with open("video_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("\nData byla uložena do 'video_analysis.json'.")
    return num_videos, video_lengths, stanice_info


# Spusť analýzu
video_length_info = video_length_analysis()