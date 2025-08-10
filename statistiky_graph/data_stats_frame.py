import json
import xml.etree.ElementTree as ET

# Sem si dosaď své třídy
CLASSES = ['studio', 'indoor', 'outdoor', 'předěl', 'reklama', 'upoutávka', 'grafika', 'zábava']

def get_eaf(eaf_file: str):
    """
    Read an EAF file and return annotations.
    Each annotation includes: start_time (ms), end_time (ms), class label, time_slot_ref1, time_slot_ref2
    """
    tree = ET.parse(eaf_file)
    root = tree.getroot()

    # Get time values
    time_slot_list = [time_slot.attrib for time_slot in root.find('TIME_ORDER')]
    time_order = {slot['TIME_SLOT_ID']: int(slot['TIME_VALUE']) for slot in time_slot_list}

    annotations = []
    for tier in root.findall('TIER'):
        for annotation in tier.findall('ANNOTATION/ALIGNABLE_ANNOTATION'):
            ref1 = annotation.attrib['TIME_SLOT_REF1']
            ref2 = annotation.attrib['TIME_SLOT_REF2']
            value = annotation.find('ANNOTATION_VALUE').text
            if value is not None:
                annotations.append((time_order[ref1], time_order[ref2], value.strip(), ref1, ref2))
    return annotations

def analyze_min_instances(meta_file, fps=25):
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)

    # Inicializace minima pro každou třídu
    min_class_instances = {
        cls: {
            'length': float('inf'),
            'start_time': None,
            'end_time': None,
            'class': cls,
            'video': None,
            'timeslot_ids': (None, None)
        } for cls in CLASSES
    }

    for entry in meta_data:
        annotation_file = entry['annotation']
        video_name = entry['video']

        try:
            annotations = get_eaf(annotation_file)
        except Exception as e:
            print(f"Chyba při zpracování {annotation_file}: {e}")
            continue

        for (start_time, end_time, cls, ref1, ref2) in annotations:
            if cls not in CLASSES:
                continue

            length = end_time - start_time

            if length < min_class_instances[cls]['length']:
                min_class_instances[cls] = {
                    'length': length,
                    'start_time': start_time,
                    'end_time': end_time,
                    'class': cls,
                    'video': video_name,
                    'timeslot_ids': (ref1, ref2)
                }

    # Výpis výsledků
    for cls, data in min_class_instances.items():
        print(f"Třída: {cls}")
        print(f"Nejkratší výskyt: {data['length']} ms ({data['length'] / 1000:.2f} s)")
        print(f"Čas: {data['start_time']} – {data['end_time']} ms")
        print(f"TimeSlot ID: {data['timeslot_ids'][0]} → {data['timeslot_ids'][1]}")
        print(f"Video: {data['video']}")
        print()


analyze_min_instances("RAVDAI/metadata/metadata_local.json", fps=25)