import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from Augmentation import audio_to_melspectrogram

# -----------------------------
# Load test.csv
# -----------------------------
test_csv = "test.csv"
df = pd.read_csv(test_csv)

X_test = []
y_test = []

for idx, row in df.iterrows():
    audio_path = row['audio_path']
    label = row['label']
    mel_spec = audio_to_melspectrogram(audio_path)
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    X_test.append(mel_spec)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y_test)
y_enc = to_categorical(y_enc)

# Load model
model = load_model("cough_model.h5")

# Evaluate
loss, acc = model.evaluate(X_test, y_enc)
print(f"Test Accuracy: {acc*100:.2f}%")
