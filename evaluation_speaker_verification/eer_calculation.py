import argparse
import os
import pickle as pkl
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
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

    return np.asarray(all_scores), np.asarray(all_keys)


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

    args = parser.parse_args()
    trial_file, emb, save_kaldi_emb = args.trial_file, args.emb, args.save_kaldi_emb

    y_score, y = get_acc(trial_file=trial_file, emb=emb, save_kaldi_emb=save_kaldi_emb)
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    sys.stdout.write("{0:.2f}\n".format(eer * 100))
