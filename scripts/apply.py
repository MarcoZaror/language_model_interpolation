# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from aux_functions import combine_pbb, compute_perplexity, files_in
from utils import read_probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_weights",
        type=str,
        help="Path to weights files.",
    )
    parser.add_argument(
        "--probs_eval",
        type=str,
        help="Path to evaluation probabilities files.",
    )
    args = parser.parse_args()

    weights_path = Path(args.path_weights)
    lambdas_list = np.loadtxt(weights_path)
    files_eval = files_in(args.probs_eval)

    evaluation = []
    for file in files_eval:
        text = read_probabilities(file)
        evaluation.append(text)

    loss_eva = []
    for i in range(lambdas_list.shape[0]):
        lambdas = lambdas_list[i, :]
        evaluation_int = combine_pbb(lambdas, evaluation)
        ppl_eva, words = compute_perplexity(evaluation_int)
        loss_eva.append(ppl_eva)
        print("Evaluation: Loss {:.2f}, using {} words".format(ppl_eva, words))

    plt.title("Perplexity computed on evaluation set")
    plt.plot(loss_eva)
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity")
    plt.show()
