import argparse
import os
import pickle as pkl
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_acc(trial_file='', emb='', save_kaldi_emb=False):
    trial_score = open('trial_score.txt', 'w')
    dirname = os.path.dirname(trial_file)
    
    with open(emb, 'rb') as f:
        emb = pkl.load(f)
        
    trial_embs = []
    keys = []
    all_scores = []
    all_keys = []

    with open(trial_file, 'r') as f:
        tmp_file = f.readlines()
        for line in tqdm(tmp_file):
            line = line.strip()
            truth, x_speaker, y_speaker = line.split()

            x_speaker = x_speaker.split('/')
            x_speaker = '@'.join(x_speaker)

            y_speaker = y_speaker.split('/')
            y_speaker = '@'.join(y_speaker)

            X = emb[x_speaker]
            Y = emb[y_speaker]

            if save_kaldi_emb and x_speaker not in keys:
                keys.append(x_speaker)
                trial_embs.extend([X])

            if save_kaldi_emb and y_speaker not in keys:
                keys.append(y_speaker)
                trial_embs.extend([Y])

            score = np.dot(X, Y) / ((np.dot(X, X) * np.dot(Y, Y)) ** 0.5)
            score = (score + 1) / 2

            all_scores.append(score)
            trial_score.write(str(score) + "\t" + truth + "\n")
            all_keys.append(int(truth))

    trial_score.close()

    if save_kaldi_emb:
        np.save(os.path.join(dirname, 'all_embs_voxceleb.npy'), np.asarray(trial_embs))
        np.save(os.path.join(dirname, 'all_ids_voxceleb.npy'), np.asarray(keys))
        print("Saved KALDI PLDA related embeddings to {}".format(dirname))

    return np.asarray(all_scores), np.asarray(all_keys), emb


def plot_eer_curve(y, y_score, output_path):
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    far = fpr
    frr = 1 - tpr

    eer_index = np.nanargmin(np.abs(far - frr))
    eer_threshold = thresholds[eer_index]
    eer = far[eer_index]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, far * 100, label="False Accept Rate (FAR)", color="red")
    plt.plot(thresholds, frr * 100, label="False Reject Rate (FRR)", color="blue")

    plt.axvline(eer_threshold, color="black", linestyle="dashed", label="Equal Error Rate (EER)")
    plt.scatter([eer_threshold], [eer * 100], color="black", zorder=5)
    plt.text(eer_threshold, eer * 100 + 2, f"EER = {eer * 100:.2f}%", fontsize=10,
             verticalalignment='bottom', horizontalalignment='center')

    plt.xlabel("Decision Threshold / Sensitivity")
    plt.ylabel("Error Rate (%)")
    plt.title("Equal Error Rate (EER) Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def plot_mindcf(y, y_score, output_path, p_target=0.01, c_miss=1, c_fa=1):
    """
    Calculate and plot minimum Detection Cost Function (minDCF).
    p_target, c_miss, c_fa are parameters for DCF calculation.
    """

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    fnr = 1 - tpr  # False Negative Rate
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)

    min_dcf_idx = np.argmin(dcf)
    min_dcf = dcf[min_dcf_idx]
    min_dcf_threshold = thresholds[min_dcf_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, dcf, label="Detection Cost Function (DCF)")
    plt.axvline(min_dcf_threshold, color='black', linestyle='dashed', label=f"minDCF = {min_dcf:.4f}")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Detection Cost Function")
    plt.title("Minimum Detection Cost Function (minDCF)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    return min_dcf


def plot_tsne(emb, keys, output_path):
    # Assuming emb is numpy array [N x dim], keys is list of speaker IDs
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30, n_iter=1000)
    emb_2d = tsne.fit_transform(emb)

    # Map each speaker ID to a color
    unique_speakers = list(set(keys))
    num_speakers = len(unique_speakers)
    from matplotlib import colormaps
    cmap = colormaps.get_cmap("tab20", num_speakers)

    speaker_to_color = {spk: cmap(i) for i, spk in enumerate(unique_speakers)}
    colors = [speaker_to_color[k] for k in keys]

    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=10, alpha=0.7)
    plt.title("t-SNE Visualization of Speaker Embeddings")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_file", help="Path to VoxCeleb trial file", type=str, required=True)
    parser.add_argument("--emb", help="Path to pickle file of embeddings dictionary", type=str, required=True)
    parser.add_argument(
        "--save_kaldi_emb",
        help="Save kaldi embeddings for KALDI PLDA training later",
        required=False,
        action='store_true'
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save all plots",
        type=str,
        required=True
    )

    args = parser.parse_args()
    trial_file, emb_path, save_kaldi_emb, output_dir = args.trial_file, args.emb, args.save_kaldi_emb, args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y_score, y, emb_dict = get_acc(trial_file=trial_file, emb=emb_path, save_kaldi_emb=save_kaldi_emb)

    # Plot EER curve
    eer_plot_path = os.path.join(output_dir, "eer_curve.png")
    plot_eer_curve(y, y_score, eer_plot_path)
    print(f"EER curve saved to: {eer_plot_path}")

    # Calculate EER for printing
    fpr, tpr, _ = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    print(f"EER: {eer * 100:.2f}%")

    # Plot minDCF
    mindcf_plot_path = os.path.join(output_dir, "mindcf_curve.png")
    min_dcf = plot_mindcf(y, y_score, mindcf_plot_path)
    print(f"minDCF: {min_dcf:.4f} (plot saved to {mindcf_plot_path})")

    # Plot t-SNE visualization
    if save_kaldi_emb:
        emb_path_for_tsne = os.path.join(os.path.dirname(trial_file), 'all_embs_voxceleb.npy')
        keys_path = os.path.join(os.path.dirname(trial_file), 'all_ids_voxceleb.npy')

        if os.path.exists(emb_path_for_tsne) and os.path.exists(keys_path):
            emb_array = np.load(emb_path_for_tsne)
            keys = np.load(keys_path, allow_pickle=True)
            tsne_plot_path = os.path.join(output_dir, "tsne_plot.png")
            plot_tsne(emb_array, keys, tsne_plot_path)
            print(f"t-SNE plot saved to: {tsne_plot_path}")
        else:
            print("KALDI embeddings or keys not found, skipping t-SNE plot.")
    else:
        print("Enable --save_kaldi_emb flag to save embeddings for t-SNE plot.")

