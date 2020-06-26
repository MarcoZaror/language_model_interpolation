# language_model_interpolation

This solution consists of 3 files:

### File: estimate.py
 - Objective: Estimate weights for a given number of sequences of n-gram probrabilities
 - Input: 
	- Path to sequences of n-gram probabilities
	- Name of the file where to store the learned weights 
 - Output:
	- The learned weights are stored in the same directory
 - Command to run: python estimate.py <folder sequences of n-grams> <name of file to store learned weights>
 - Example: python estimate.py dev/*.probs weights


### File: apply.py
 - Objective: Take a sequence of n-gram probabilities and compute the perplexity of the interpolated set
 - Input: 
	- Name of the file containing the learned weights
	- Path to sequences of n-gram probabilities
 - Command to run: python evaluate.py <name of the file with learned weights> <folder sequences of n-grams>
 - Example: python apply.py weights eval/*.probs


### File: aux_functions.py
 - Objective: Provide functions to preprocess data, compute perplexity and optimization
