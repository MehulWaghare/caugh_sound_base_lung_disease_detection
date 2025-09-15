import sounddevice as sd
import soundfile as sf
from predict import predict_disease

def record_cough(duration=3, sr=22050, filename="user_cough.wav"):
    print("🎤 Recording cough for 3 seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(filename, audio, sr)
    print(f"✅ Saved recording as {filename}")
    return filename

if __name__ == "__main__":
    audio_file = record_cough()
    disease = predict_disease(audio_file)
    print(f"Predicted Disease: {disease}")
