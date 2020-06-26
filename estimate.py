# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:15:28 2020

@author: Marco
OBJECTIVE: Learn interpolation weights from probabilities of different LMs

DESCRIPTION: This file receive L provided sequences of n-gram probabilities and return 
a set of interpolated weights for each iteration of the optimization
process (Last set is the optimal)

INPUT: Development data (robabilities from different language models)
OUTPUT: Learned interpolation weights   
    
USE: python <py_file> <dev data> <out file>
EXAMPLE: python estimate.py dev/*.probs weigths.txt
"""
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from aux_functions import files_in, computePPL, combine_pbb, update_lambda, optimization

#def files_in(path): 
#    return glob.glob(path)

if len(sys.argv) > 1:
    files_dev = files_in(sys.argv[1])
else:
    files_dev = files_in('dev/*.probs')

development = []
for file in files_dev:
    with open(file, 'r') as f:
        text = f.read()
        text = text.split('\n')
        text = [float(x) for x in text if x != '']
        development.append(text)

'''files = files_in('eval/*.probs')
evaluation = []
for file in files:
    with open(file, 'r') as f:
        text = f.read()
        text = text.split('\n')
        text = [float(x) for x in text if x != '']
        evaluation.append(text)'''
    
development = np.asarray(development, dtype = np.float64)
#evaluation = np.asarray(evaluation, dtype = np.float64)


l,loss_dev, ls = optimization(development)

plt.title('Perplexity computed on development set')
plt.plot(loss_dev)#, label = 'Development')
plt.xlabel('Iterations')
plt.ylabel('Perplexity')
#plt.legend()
plt.show()

'''
#Analysis
max_value = l.argmax()
print(files_dev[max_value])
idxs = np.argsort(l)
print(l*100)

x = [x[4:-6] for x in files_dev]
plt.bar(x, l*100)
plt.ylabel('Weights')
plt.xlabel('Resources')
plt.title('Distribution of learnt weights')

print(files_dev)
[x for _,x in sorted(zip(l,files_dev),reverse = True)]
'''

ls = np.array(ls)
if len(sys.argv) > 1:
    np.savetxt(sys.argv[2], ls, newline="\n")
else:
 np.savetxt("weights.txt", ls, newline="\n")

