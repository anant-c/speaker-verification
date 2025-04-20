import os
import wave
import contextlib
from pydub import AudioSegment

def split_wav_files(base_dir):
    chunk_base_dir = os.path.join(base_dir, "chunk")
    os.makedirs(chunk_base_dir, exist_ok=True)
    
    # Iterate through all label directories in the base directory
    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)
        if not os.path.isdir(label_path) or label == "chunk":
            continue
        
        label_chunk_dir = os.path.join(chunk_base_dir, label)
        os.makedirs(label_chunk_dir, exist_ok=True)
        
        for filename in os.listdir(label_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(label_path, filename)
                audio = AudioSegment.from_wav(file_path)
                duration_ms = len(audio)
                chunk_size = 3000  # 3 seconds in milliseconds
                
                # Split and save chunks
                for i, start in enumerate(range(0, duration_ms, chunk_size)):
                    end = min(start + chunk_size, duration_ms)
                    chunk = audio[start:end]
                    chunk_filename = f"{os.path.splitext(filename)[0]}_chunk{i+1}.wav"
                    chunk.export(os.path.join(label_chunk_dir, chunk_filename), format="wav")

if __name__ == "__main__":
    base_directory = "../TEST"  # Adjust if necessary
    split_wav_files(base_directory)

