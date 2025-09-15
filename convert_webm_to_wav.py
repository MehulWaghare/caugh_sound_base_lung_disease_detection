# convert_webm_to_wav.py

import os
import subprocess
import pandas as pd

# --- Step 1: Convert .webm to .wav ---
input_dir = r"D:\caugh_sound\COUGHVID\public_dataset\public_dataset"
output_dir = r"D:\caugh_sound\COUGHVID\public_dataset\public_dataset_wav"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".webm"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace(".webm", ".wav"))
        command = f'ffmpeg -y -i "{input_path}" "{output_path}"'
        subprocess.run(command, shell=True)

print("✅ All .webm files converted to .wav!")

# --- Step 2: Update CSV files ---
csv_files = ["train.csv", "test.csv", "labels.csv"]

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # Replace .webm with .wav in audio_path
    df["audio_path"] = df["audio_path"].str.replace(".webm", ".wav")
    df.to_csv(csv_file, index=False)
    print(f"✅ Updated {csv_file}")
