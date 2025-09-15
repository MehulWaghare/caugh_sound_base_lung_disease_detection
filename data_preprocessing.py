import pandas as pd
from sklearn.model_selection import train_test_split
import os

# -----------------------------
# Paths
# -----------------------------
csv_path = r"D:\caugh_sound\COUGHVID\public_dataset\public_dataset\metadata_compiled.csv"
audio_dir = r"D:\caugh_sound\COUGHVID\public_dataset\public_dataset"

# -----------------------------
# Load metadata
# -----------------------------
meta = pd.read_csv(csv_path)

# Keep only necessary columns
df = meta[["uuid", "status", "respiratory_condition", "cough_detected"]].copy()

# -----------------------------
# Build audio paths
# -----------------------------
df["audio_path"] = df["uuid"].apply(lambda x: os.path.join(audio_dir, x + ".webm"))

# Keep only rows where audio file exists
df = df[df["audio_path"].apply(os.path.exists)]

# -----------------------------
# Map labels
# -----------------------------
def map_label(row):
    status = str(row["status"]).lower().strip() if pd.notna(row["status"]) else ""
    resp = str(row["respiratory_condition"]).lower().strip() if pd.notna(row["respiratory_condition"]) else ""

    if status in ["healthy", "normal"]:
        return "healthy"
    elif status in ["covid-19", "covid", "positive_mild", "positive_moderate", "positive_asymp"]:
        return "covid"
    elif status == "symptomatic":
        if resp == "true":
            return "respiratory_disease"
        else:
            return "symptomatic"
    else:
        return None  # unknown or missing

df["label"] = df.apply(map_label, axis=1)

# Remove rows with unknown labels
df = df[df["label"].notna()]

# -----------------------------
# Optional: include weak coughs (threshold 0.02)
# -----------------------------
df = df[df["cough_detected"] >= 0.02]

# -----------------------------
# Check for empty DataFrame
# -----------------------------
if df.empty:
    raise ValueError("❌ No valid data found after filtering. Check audio paths, status values, or cough_detected thresholds!")

# -----------------------------
# Save all labels
# -----------------------------
df[["uuid", "audio_path", "label"]].to_csv("labels.csv", index=False)

# -----------------------------
# Train-test split
# -----------------------------
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

train_df[["uuid", "audio_path", "label"]].to_csv("train.csv", index=False)
test_df[["uuid", "audio_path", "label"]].to_csv("test.csv", index=False)

# -----------------------------
# Summary
# -----------------------------
print("Class distribution:\n", df["label"].value_counts())
print(f"✅ Saved {len(train_df)} train samples and {len(test_df)} test samples.")
