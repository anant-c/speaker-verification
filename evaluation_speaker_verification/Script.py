import os
import pickle
import argparse
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from omegaconf import OmegaConf
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

def load_titanet():
    """Load the Titanet speaker embedding model."""
    #EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
    model = EncDecSpeakerLabelModel.restore_from("./titanet-large-IMSV.nemo")
    model.eval()
    return model

def extract_embedding(model, audio_path):
    """Extract speaker embedding from an audio file using the correct API."""
    with torch.no_grad():
        embedding = model.get_embedding(audio_path).cpu().numpy().flatten()
    return embedding

def process_voxceleb(dataset_dir, model, output_pickle, output_txt):
    """Process all audio files in the Dataset directory and store embeddings."""
    embeddings = {}
    
    with open(output_txt, "w") as txt_file:
        for root, dirs, files in os.walk(dataset_dir):
            if files:  # Only process if there are audio files
                speaker_id = os.path.basename(os.path.dirname(root))  # Get the speaker ID from the parent directory
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


def get_acc(trial_file, emb_file, save_kaldi_emb=False):
    """Calculate accuracy and EER using cosine similarity."""
    trial_score = open('trial_score.txt', 'w')
    dirname = os.path.dirname(trial_file)
    
    with open(emb_file, 'rb') as f:
        emb = pickle.load(f)
    
    trial_embs = []
    keys = []
    all_scores = []
    all_keys = []
    
    with open(trial_file, 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            truth, x_speaker, y_speaker = line.split()

            x_speaker = x_speaker.replace("/", "@")
            y_speaker = y_speaker.replace("/", "@")
            
            if x_speaker not in emb or y_speaker not in emb:
                print(f"Missing embeddings for: {x_speaker} or {y_speaker}")
                continue
            
            X, Y = emb[x_speaker], emb[y_speaker]

            if save_kaldi_emb and x_speaker not in keys:
                keys.append(x_speaker)
                trial_embs.append(X)
            if save_kaldi_emb and y_speaker not in keys:
                keys.append(y_speaker)
                trial_embs.append(Y)
            
            score = np.dot(X, Y) / ((np.dot(X, X) * np.dot(Y, Y)) ** 0.5)
            score = (score + 1) / 2
            
            all_scores.append(score)
            all_keys.append(int(truth))
            trial_score.write(f"{score}\t{truth}\n")
    
    trial_score.close()
    
    if save_kaldi_emb:
        np.save(os.path.join(dirname, 'all_embs_voxceleb.npy'), np.asarray(trial_embs))
        np.save(os.path.join(dirname, 'all_ids_voxceleb.npy'), np.asarray(keys))
        print(f"Saved KALDI PLDA related embeddings to {dirname}")
    
    return np.asarray(all_scores), np.asarray(all_keys)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def plot_eer_curve(y_score, y):
    """Plot Equal Error Rate (EER) Curve."""
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    
    # Compute False Accept Rate (FAR) and False Reject Rate (FRR)
    far = fpr
    frr = 1 - tpr

    # Find the EER where FAR and FRR are closest
    eer_threshold = thresholds[np.nanargmin(np.abs(far - frr))]
    eer = far[np.nanargmin(np.abs(far - frr))]

    # Plot FAR and FRR
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far * 100, label="False Accept Rate (FAR)", color="red")
    plt.plot(thresholds, frr * 100, label="False Reject Rate (FRR)", color="blue")
    
    # Plot the EER point at the intersection
    plt.axvline(eer_threshold, color="black", linestyle="dashed", label="Equal Error Rate (EER)")
    plt.scatter([eer_threshold], [eer * 100], color="black", zorder=3)
    plt.text(eer_threshold, eer * 100 + 2, f"EER = {eer * 100:.2f}%", fontsize=10, verticalalignment='bottom')

    plt.xlabel("Decision Threshold / Sensitivity")
    plt.ylabel("Errors (%)")
    plt.title("Equal Error Rate (EER) Curve")
    plt.legend()
    plt.grid()
    plt.show()

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", help="Path to VoxCeleb directory", type=str, required=True)
    parser.add_argument("--trial_file", help="Path to VoxCeleb trial file", type=str, required=True)
    parser.add_argument("--output_pickle", help="Path to save embeddings pickle file", type=str, default="speaker_embeddings.pkl")
    parser.add_argument("--output_txt", help="Path to save embeddings text file", type=str, default="speaker_embeddings.txt")
    parser.add_argument("--save_kaldi_emb", help="Save kaldi embeddings for KALDI PLDA training", action='store_true')
    
    args = parser.parse_args()
    
    print("Loading Titanet model...")
    model = load_titanet()
    
    print("Processing dataset...")
    process_voxceleb(args.dataset_dir, model, args.output_pickle, args.output_txt)
    
    print("Calculating accuracy and EER...")
    y_score, y = get_acc(trial_file=args.trial_file, emb_file=args.output_pickle, save_kaldi_emb=args.save_kaldi_emb)
    
    fpr, tpr, _ = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    print(f"Equal Error Rate (EER): {eer * 100:.2f}%")
    
    plot_eer_curve(y_score, y)

if __name__ == "__main__":
    main()