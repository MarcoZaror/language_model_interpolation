### Objective
This project is focused on finding an optimal combination of probabilities from different language models in otder to produce a stronger set of probabilities.

### Methodology
The proposed approach is a 3-step methodology

1. Initialize a set of lambdas (combination weights)
2. Use lambdas to combine probabilities and compute perplexity in an evaluation dataset
3. Compute gradient and update lambdas accordingly 
4. Repeat from 2

### Scripts

estimate.py
 - Objective: Estimate interpolation weights from probabilities of different language models
 - Example: python estimate.py dev/*.probs weights.probs

apply.py
 - Objective: Apply the learned lambdas (weights) to an evaluation set
 - Example: python apply.py weights.txt eval/*.probs
