
import random
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from pathlib import Path
import numpy as np


# === Paths ===
base_dir = "data/Experiment_2"
cam0_dir = os.path.join(base_dir, "exp1_cam0")
cam1_dir = os.path.join(base_dir, "exp1_cam1")

os.makedirs(cam0_dir, exist_ok=True)
os.makedirs(cam1_dir, exist_ok=True)

# === Sets to track duplicates ===
seen_cam0 = set()
seen_cam1 = set()

# === Index counters ===
index_cam0 = 1
index_cam1 = 1

# === Helper Function ===
def process_image(camera_prefix, rotation_angle, seen_set, index, filepath, dst_dir):
    timestamp = os.path.splitext(os.path.basename(filepath))[0][len(camera_prefix):]

    if timestamp not in seen_set:
        seen_set.add(timestamp)
        try:
            with Image.open(filepath) as img:
                rotated = img.rotate(rotation_angle, expand=True)
                new_filename = f"{camera_prefix}{timestamp}({index}).jpg"
                rotated.save(os.path.join(dst_dir, new_filename))
            index += 1
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return index, False
        return index, True
    return index, False

# === Process images ===
for filename in sorted(os.listdir(base_dir)):
    if not filename.lower().endswith((".jpg", ".jpeg")):
        continue

    filepath = os.path.join(base_dir, filename)

    if filename.startswith("cam0_"):
        index_cam0, processed = process_image(
            "cam0_", -90, seen_cam0, index_cam0, filepath, cam0_dir
        )
        if processed:
            os.remove(filepath)

    elif filename.startswith("cam1_"):
        index_cam1, processed = process_image(
            "cam1_", 90, seen_cam1, index_cam1, filepath, cam1_dir
        )
        if processed:
            os.remove(filepath)