import librosa
import numpy as np

def extract_features(file_path, sr=16000, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return np.zeros((n_mfcc,))
