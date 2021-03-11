#############################################################################################################

# Written by Eyal Betzalel 20.1.2021

#############################################################################################################

# In this section I will analyze samples with different divergence functions : 

# Inputs :
# 1. Unified table with dataset and nll score for each input from different models
# 2. --f (kl / js / ot / more?)

# Output : 
#1. score

#############################################################################################################

from LoadData import unified_table
import numpy as np

def kl_divergence(p, q):
	return (1.0/len(p)) * sum(np.log2(p[i]/q[i]) for i in range(len(p)))

p = []
q = []

for i in range(len(unified_table)):
    p.append((np.exp(-1 * unified_table[i][1]))/1024)
    q.append((np.exp(-1 * unified_table[i][2]))/1024)


yos = kl_divergence(p, q)

v=0