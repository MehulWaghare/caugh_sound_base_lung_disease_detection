# predict.py

import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from Augmentation import audio_to_melspectrogram

# 1️⃣ Load the trained model
print("⏳ Loading model...")
model = load_model("cough_detection_model.h5")
print("✅ Model loaded successfully!")

# 2️⃣ Load Label Encoder
with open("le.pkl", "rb") as f:
    le = pickle.load(f)
print("✅ LabelEncoder loaded!")

# 3️⃣ Open File Dialog to choose audio
root = tk.Tk()
root.withdraw()  # hide main window
file_path = filedialog.askopenfilename(
    title="Select a Cough Sound File",
    filetypes=[("Audio Files", "*.wav *.webm *.mp3")]
)

if not file_path:
    messagebox.showerror("Error", "❌ No file selected. Exiting.")
    exit()

print(f"🎵 Selected file: {file_path}")

# 4️⃣ Convert to Mel-Spectrogram
try:
    mel_spec = audio_to_melspectrogram(file_path)
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # add channel dimension
    mel_spec = np.expand_dims(mel_spec, axis=0)   # add batch dimension
except Exception as e:
    messagebox.showerror("Error", f"❌ Failed to process file: {e}")
    exit()

# 5️⃣ Predict
pred = model.predict(mel_spec)
pred_class = np.argmax(pred, axis=1)[0]
pred_label = le.inverse_transform([pred_class])[0]

# 6️⃣ Show Result
messagebox.showinfo("Prediction Result", f"🩺 Predicted Disease: {pred_label}")
print(f"🩺 Predicted Disease: {pred_label}")
