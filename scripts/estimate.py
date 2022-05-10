# -*- coding: utf-8 -*-

import argparse

import matplotlib.pyplot as plt
import numpy as np
from aux_functions import files_in, optimization
from utils import read_probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probs_development",
        type=str,
        help="Path to development probabilities files.",
    )
    parser.add_argument(
        "--path_weights_store",
        type=str,
        help="Path to store weights.",
    )
    args = parser.parse_args()

    files_dev = files_in(args.probs_eval)

    development = []
    for file in files_dev:
        with open(file, "r") as f:
            text = read_probabilities(file)
            development.append(text)

    development = np.asarray(development, dtype=np.float64)

    # Do optimization
    l, loss_dev, ls = optimization(development)

    plt.title("Perplexity computed on development set")
    plt.plot(loss_dev)
    plt.xlabel("Iterations")
    plt.ylabel("Perplexity")
    plt.show()

    ls = np.array(ls)
    np.savetxt(args.path_weights_store, ls, newline="\n")
