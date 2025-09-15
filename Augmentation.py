# Augmentation.py
import numpy as np
from pydub import AudioSegment
import librosa

def audio_to_melspectrogram(file_path, sr=22050, n_mels=64, max_len=100):
    """
    Convert audio (.wav or .webm) to mel-spectrogram with fixed length
    """
    try:
        # load audio
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= np.max(np.abs(samples))  # normalize

        # mel-spectrogram
        mel = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=n_mels)
        mel = librosa.power_to_db(mel, ref=np.max)

        # pad or truncate to max_len
        if mel.shape[1] < max_len:
            pad_width = max_len - mel.shape[1]
            mel = np.pad(mel, ((0,0),(0,pad_width)), mode='constant')
        else:
            mel = mel[:, :max_len]

        return mel
    except Exception as e:
        raise ValueError(f"Error processing {file_path}: {e}")

def add_noise(mel, noise_factor=0.005):
    return mel + noise_factor * np.random.randn(*mel.shape)

def time_shift(mel, shift_max=5):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(mel, shift, axis=1)
