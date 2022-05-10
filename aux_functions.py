# -*- coding: utf-8 -*-
import glob
import logging
from typing import List

import numpy as np

logger = logging.getLogger("language_model_interpolation")


def files_in(path: str):
    """
    Get all files in path
    Example path: 'eval/*.probs' -> All files in folder eval with a .probs extension
    """
    return glob.glob(path)


def compute_perplexity(dataset: np.ndarray) -> (float, int):
    epsilon = 1e-10
    log_dataset = np.log(dataset + epsilon)
    vocab_size = dataset.shape[0]
    return np.exp((-1 / vocab_size) * np.sum(log_dataset)), vocab_size


def combine_pbb(lambdas: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    """Combine probabilities from different language models"""
    return np.dot(lambdas.T, dataset)


def update_lambda(lambdas: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    """Update lambdas given dataset"""
    result = np.zeros_like(dataset)
    epsilon = 1e-10
    den = combine_pbb(lambdas, dataset) + epsilon
    for i in range(len(lambdas)):
        pbb = lambdas[i] * dataset[i]
        result[i] = pbb / den
    updated_lamb = np.sum(result, axis=1)
    updated_lamb = updated_lamb / dataset.shape[1]
    updated_lamb = updated_lamb[:-1]
    lambda_10 = 1 - np.sum(updated_lamb)
    lambdas_f = np.zeros_like(lambdas)
    lambdas_f[:9] = updated_lamb
    lambdas_f[9] = lambda_10
    return lambdas_f


def optimization(
    development: np.ndarray, init_strategy: str = "random"
) -> (np.ndarray, float, List):
    """
    Compute lambdas that will provide the best combinations of language
    model probabilities
    init_strategy: Initialization strategy for the lambdas. Three values
    accepted at the moment: random, uniform
    """
    # Initialize lambdas
    ls = []
    if init_strategy == "random":
        lambdas = np.random.random(10)
        lambdas /= lambdas.sum()
    elif init_strategy == "uniform":
        lambdas = np.ones(10)
        lambdas = lambdas / 10
    else:
        logger.error(f"Init_strategy: {init_strategy} not accepted")
    ls.append(lambdas)

    diff = 1
    loss_dev = []
    while np.abs(diff) > 0.1:
        # Apply lambdas and compute probabilities
        development_int = combine_pbb(lambdas, development)

        # Compute perplexity using probabilities and evaluate them
        ppl_dev, words = compute_perplexity(development_int)
        loss_dev.append(ppl_dev)

        print("Development: Loss {:.2f}, using {} words".format(ppl_dev, words))
        # Update lambdas (Compute posterior and then update lambda)
        lambdas = update_lambda(lambdas, development)
        ls.append(lambdas)

        # Stopping criteria
        if len(loss_dev) > 1:
            diff = loss_dev[-1] - loss_dev[-2]
    return lambdas, loss_dev, ls
