import argparse
from inception_score import inception_score
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
import time

# args :

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    type=str,
    help="Path to ImageGPT results",
    default=r"/home/dsi/eyalbetzalel/pytorch-generative-v6/tmp/run/ep_0_ch_128_psb_2_resb_4_atval_64_attk_8/",
)

args = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

dataset = datasets.ImageFolder(args.path, transform=transform)
mean_split_scores_arr = []
for k in range(len(dataset)):

    idx = [i for i in range(len(dataset)) if i != k]
    # build the appropriate subset
    subset = Subset(dataset, idx)
    print('iter = ', k)
    t = time.time()
    mean_split_scores, std_split_scores = inception_score(subset, cuda=True, batch_size=32, resize=True, splits=1)
    elapsed = time.time() - t
    print('elapsed = ', elapsed)
    mean_split_scores_arr.append(mean_split_scores_arr)
    if k == 3:
        break

# import ipdb; ipdb.set_trace()

mean_split_scores_arr_np = np.asarray(mean_split_scores_arr)
# xi = [0] * len(mean_split_scores_arr)

xi = [0] * 4

# for i in range(len(mean_split_scores_arr)):
for i in range(4):
    xi[i] = np.sum(np.delete(mean_split_scores_arr_np, i))

# Step 2 - Calculate Mean and VAR :


# mu = (1.0 / len(mean_split_scores_arr_np)) * sum((xi[i]) for i in range(len(mean_split_scores_arr_np)))
mu = (1.0 / 4) * sum((xi[i]) for i in range(4))

# var = ((len(mean_split_scores_arr_np)-1) / len(mean_split_scores_arr_np)) * sum(((xi[i]-mu) ** 2) for i in range(len(mean_split_scores_arr_np)))

var = (3.0 / 4) * sum((xi[i] - mu) ** 2)
for i in range(4)
print("Calculating Inception Score...")
print(mu)
print(var)
