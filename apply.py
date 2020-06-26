# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:51:49 2020

@author: Marco
OBJECTIVE: Apply the learned weights to evaluation data

DESCRIPTION: This file receive L provided sequences of n-gram probabilities for evaluation
and the intermediate weights. Using that, it provides the perplexity for every
set of weights.

INPUT: Learned interpolation weights, evaluation data
    
USE: python <py_file> <weights> <eval data>
EXAMPLE: python apply.py weights.txt eval/*.probs
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from aux_functions import files_in, computePPL, combine_pbb, update_lambda, optimization

# Import weights and evaluation data
if len(sys.argv) > 1:
    lambdas_list = np.loadtxt(sys.argv[1])
else:
    lambdas_list = np.loadtxt("weights.txt")

if len(sys.argv) > 1:
    files_eva = files_in(sys.argv[2])
else:
    files_eva = files_in('eval/*.probs')


evaluation = []
for file in files_eva:
    with open(file, 'r') as f:
        text = f.read()
        text = text.split('\n')
        text = [float(x) for x in text if x != '']
        evaluation.append(text)
        
loss_eva = []
for i in range(lambdas_list.shape[0]):
    lambdas = lambdas_list[i,:]
    evaluation_int = combine_pbb(lambdas, evaluation)
    ppl_eva,words = computePPL(evaluation_int)
    loss_eva.append(ppl_eva)
    print('Evaluation: Loss {:.2f}, using {} words'.format(ppl_eva, words))


plt.title('Perplexity computed on evaluation set')
plt.plot(loss_eva)#, label = 'Evaluation')
plt.xlabel('Iterations')
plt.ylabel('Perplexity')
#plt.legend()
plt.show()


