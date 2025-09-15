import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Augmentation import audio_to_melspectrogram, add_noise, time_shift

# Paths
train_csv = "train.csv"
test_csv = "test.csv"

# Load CSVs
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Encode labels
le = LabelEncoder()
y_all = le.fit_transform(train_df["label"])

# Save LabelEncoder for use in prediction
with open("le.pkl", "wb") as f:
    pickle.dump(le, f)
print("✅ LabelEncoder saved as le.pkl")

# ---- SAMPLE 3000 TRAINING + 600 TEST SAMPLES ----
train_small, _, y_train_small, _ = train_test_split(
    train_df,
    y_all,
    train_size=3000,
    stratify=y_all,
    random_state=42
)

test_small, _, y_test_small, _ = train_test_split(
    test_df,
    le.transform(test_df["label"]),
    train_size=600,
    stratify=test_df["label"],
    random_state=42
)

print(f"✅ Using {len(train_small)} training samples and {len(test_small)} test samples")

# One-hot encode labels
y_train_cat = to_categorical(y_train_small)
y_test_cat = to_categorical(y_test_small)

# Feature extraction
def extract_features(df):
    X = []
    kept_indices = []
    for idx, row in df.iterrows():
        audio_path = row["audio_path"]
        try:
            mel_spec = audio_to_melspectrogram(audio_path)
            mel_spec = add_noise(mel_spec)
            mel_spec = time_shift(mel_spec)
            X.append(mel_spec)
            kept_indices.append(idx)
        except Exception as e:
            print(f"⚠️ Skipped {audio_path}: {e}")
    X = np.array(X)
    X = X[..., np.newaxis]  # Add channel dimension for CNN
    return X, kept_indices

print("⏳ Extracting features from training data...")
X_train, kept_train_idx = extract_features(train_small)
y_train_cat = y_train_cat[[train_small.index.get_loc(i) for i in kept_train_idx]]

print("⏳ Extracting features from test data...")
X_test, kept_test_idx = extract_features(test_small)
y_test_cat = y_test_cat[[test_small.index.get_loc(i) for i in kept_test_idx]]

print(f"✅ Feature shapes: {X_train.shape}, {X_test.shape}")

# Define CNN model
input_shape = X_train.shape[1:]

model = Sequential([
    Input(shape=input_shape),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train (10 epochs for quick test run)
print("🚀 Starting training...")
model.fit(
    X_train,
    y_train_cat,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    verbose=1
)

# Save model
model.save("cough_detection_model.h5")
print("✅ Model saved as cough_detection_model.h5")
