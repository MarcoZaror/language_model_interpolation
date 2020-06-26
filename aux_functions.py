# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:19:57 2020

@author: Marco
"""
import glob
import numpy as np

def files_in(path): 
    return glob.glob(path)

def computePPL(dataset):
    epsilon = 1e-10
    log_dataset = np.log(dataset + epsilon)
    vocab = dataset.shape[0]
    return np.exp((-1/vocab)*np.sum(log_dataset)), vocab

# Combine probabilities from different language models
def combine_pbb(lambdas, dataset):
    return np.dot(lambdas.T,dataset)

# (9)
def update_lambda(lambdas, dataset):
    result = np.zeros_like(dataset)
    epsilon = 1e-10
    den = combine_pbb(lambdas, dataset) + epsilon
    for i in range(len(lambdas)):
        pbb = lambdas[i]*dataset[i]
        result[i] = pbb/den
    updated_lamb = np.sum(result,axis=1)
    updated_lamb = updated_lamb/dataset.shape[1]
    updated_lamb = updated_lamb[:-1]
    lambda_10 = 1 - np.sum(updated_lamb)
    lambdas_f = np.zeros_like(lambdas)
    lambdas_f[:9] = updated_lamb
    lambdas_f[9] = lambda_10
    return lambdas_f

def optimization(development):
    #Initialize lambdas
    ls = []
    lambdas = np.random.random(10)
    lambdas /= lambdas.sum()
    ls.append(lambdas)
    ''' #Different initialization schemes for lambdas
    #lambdas = np.ones(10)
    #lambdas = lambdas/10
    
    #lambdas = np.array([0,0,0,0,0,0,0,0,0,1])
    '''
    
    diff = 1
    loss_dev = []
    while np.abs(diff) > 0.1:
        #Apply lambdas
        development_int = combine_pbb(lambdas, development)

        #Compute perplexity in development and evaluation data
        ppl_dev, words = computePPL(development_int)
        loss_dev.append(ppl_dev)
        
        print('Development: Loss {:.2f}, using {} words'.format(ppl_dev, words))
        
        #Update lambdas (Compute posterior and then update lambda)
        lambdas = update_lambda(lambdas, development)
        ls.append(lambdas)
        #Stop criteria
        if len(loss_dev) > 1:
            #diff = loss_eva[-1] - loss_eva[-2]
            diff = loss_dev[-1] - loss_dev[-2]
    return lambdas, loss_dev, ls

