import os
import pickle
import numpy as np
import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from omegaconf import OmegaConf
from tqdm import tqdm

def load_titanet():
    """Load the Titanet speaker embedding model."""
    #model = EncDecSpeakerLabelModel.restore_from("titanet-mera.nemo")
    model = EncDecSpeakerLabelModel.restore_from("./titanet-large-IMSV.nemo")
    model.eval()
    return model

def extract_embedding(model, audio_path):
    """Extract speaker embedding from an audio file using the correct API."""
    with torch.no_grad():
        embedding = model.get_embedding(audio_path).cpu().numpy().flatten()
    return embedding

def process_voxceleb(voxceleb_dir, model, output_pickle, output_txt):
    """Process all audio files in the VoxCeleb directory and store embeddings."""
    embeddings = {}
    
    with open(output_txt, "w") as txt_file:
        for root, dirs, files in os.walk(voxceleb_dir):
            if files:  # Only process if there are audio files
                relative_path = os.path.relpath(root, voxceleb_dir) # Get the speaker ID 
                speaker_id = relative_path.split(os.sep)[0]  
                for file in tqdm(files):
                    if file.endswith(".wav"):
                        file_path = os.path.join(root, file)
                        key = f"{speaker_id}@{file}"
                        embedding = extract_embedding(model, file_path)
                        embeddings[key] = embedding
                        
                        # Save in text format
                        txt_file.write(f"{key}: {embedding.tolist()}\n")
    
    with open(output_pickle, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {output_pickle} and {output_txt}")

if __name__ == "__main__":
    voxceleb_dir = "./segregated/English"  # Change this to your local VoxCeleb directory
    output_pickle = "speaker_embeddings-IMSV_English.pkl"
    output_txt = "speaker_embeddings-mera-IMSV_English.txt"
    
    model = load_titanet()
    process_voxceleb(voxceleb_dir, model, output_pickle, output_txt)
