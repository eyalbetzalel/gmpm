#############################################################################################################

# Written by Eyal Betzalel 20.1.2021

#############################################################################################################

# In this section I will load generated dataset with NLL values and return unified table for all of them 

# Inputs :  
# 1. ImageGPT file path
# 2. PixelSnail file path
# 3. same_model_compare_flag - if true two path are in the same format and belong to same model

# Output : 
#1. Unified table for all results

# TODO : 

#############################################################################################################

import numpy as np
import pickle
import argparse
import os

# args :

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_a",
    type=str,
    help="path to model a",
    #default=r"C:\Users\eyalb\Desktop\log\ep_0_ch_128_psb_2_resb_4_atval_64_attk_8\ep_0_ch_128_psb_2_resb_4_atval_64_attk_8_epoch_50_eval.p",
    default=r"C:\Users\eyalb\Desktop\log\ep_0_ch_128_psb_8_resb_4_atval_64_attk_8\ep_0_ch_128_psb_8_resb_4_atval_64_attk_8_epoch_50_eval.p",
)

parser.add_argument(
    "--model_b",
    type=str,
    help="path to model b",
    default=r"C:\Users\eyalb\Desktop\log\imagegpt\FullossResults_train.npy",
)

args = parser.parse_args()

def load_file(path):

    if path.endswith('.p'):
        path = open(path, "rb")
        model_table = pickle.load(path)

    elif path.endswith('.npy'):

        model_table = np.load(path)
        nll_vec = model_table[:, 1024].tolist()
        dataset = model_table[:, :1024].tolist()

        model_table = []
        for i in range(len(nll_vec)):
            model_table.append((np.asarray(dataset[i]), nll_vec[i]))

    return model_table



# Import ImageGPT files :

# TODO:

# Import PixelSnail files:

model_a = load_file(args.model_a)
model_b = load_file(args.model_b)

unified_table = []
for i in range(len(model_a)):

    data = model_a[i][0]
    nll_model_a = model_a[i][1]
    nll_model_b = model_b[i][1]
    unified_table.append((data, nll_model_a, nll_model_b))


unified_table

# Create unified table : 
