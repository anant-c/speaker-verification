import os
import glob
import torch
import torchaudio
import numpy as np
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from sklearn.metrics import roc_curve

# Step 1: Load the Titanet Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncDecSpeakerLabelModel.from_pretrained("titanet_large").to(device)

# Step 2: Load VoxCeleb Dataset
voxceleb_path = "/wav/"  # Change this to your actual dataset path
wav_files = glob.glob(os.path.join(voxceleb_path, "**", "*.wav"), recursive=True)

# Step 3: Function to Extract Speaker Embeddings
def extract_embedding(file_path):
    signal, fs = torchaudio.load(file_path)
    signal = signal.to(device)
    
    # Ensure it's a batch with a single sample
    emb = model.get_embedding(signal.unsqueeze(0)).detach().cpu().numpy()
    return emb.squeeze()

# Step 4: Generate Speaker Verification Pairs
num_samples = min(1000, len(wav_files))  # Use a subset if dataset is large
cos_sim = []
labels = []

for i in range(num_samples):
    emb1 = extract_embedding(wav_files[i])
    
    
    # Choose a second file randomly
    j = np.random.randint(0, len(wav_files))
    emb2 = extract_embedding(wav_files[j])
    
    
    # Compute cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    # # Label: 1 if same speaker, 0 otherwise (Assumption: Folder structure contains speaker IDs)
    # label = 1 if os.path.dirname(wav_files[i]) == os.path.dirname(wav_files[j]) else 0
    
    cos_sim.append(similarity)
    # labels.append(label)
print(cos_sim)
# # Step 5: Compute EER
# fpr, tpr, thresholds = roc_curve(labels, pairs,pos_label=1)
# fnr = 1 - tpr
# eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
# eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

# print(f"Equal Error Rate (EER): {eer:.4f}")
