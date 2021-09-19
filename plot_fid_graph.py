import matplotlib.pyplot as plt
import os
import numpy as np
import random

def save_res_to_file(epochs, mu, var, labels):
    rows = [epochs, mu, var]
    np.savetxt("FID" + "_" + labels + ".csv",
               rows,
               delimiter=", ",
               fmt='% s')

def plot_is_graph(epochs_vec, metrics_vec_mean, metrics_vec_var, label):
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    label = label.split('_')
    label_new = "Channels : " + label[3] + " | PS Blocks : " + label[5] + " | Res Blocks : " + label[7] + " | Attn Levels : " + label[9]
    plt.errorbar(epochs_vec[1:], metrics_vec_mean[1:],yerr=metrics_vec_var[1:], c=color, label=label_new, marker='o')


path = r"C:\Users\eyalb\Desktop\Master\Courses\Generative_Models\HW\vae\VAE\gmpm\fid_res"
res = []
curr = "0"
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        # if name == "ep_0_ch_128_psb_8_resb_4_atval_64_attk_8_epoch_15_is.out":
        #     res.append((15, 3.35, 0))
        #     continue

        epoch = int(name.split("_")[-4])
        with open(os.path.join(root, name), 'r') as file:
            data = file.read().replace('\n', ' ')
        incpetion_score_mean = float(data.split(' ')[0])
        incpetion_score_var = float(data.split(' ')[1])
        res.append((epoch, incpetion_score_mean, incpetion_score_var))
        label = "_".join(name.split("_")[:-5])
    if root != curr:
        curr = root
        print(curr)
        res_sort = sorted(res, key=lambda x: x[0])
        tuples = zip(*res_sort)
        epochs_vec, metrics_vec_mean, metrics_vec_var = [list(tuple) for tuple in tuples]
        save_res_to_file(epochs_vec, metrics_vec_mean, metrics_vec_var, label)
        plot_is_graph(epochs_vec, metrics_vec_mean, metrics_vec_var, label)
        res = []
        plt.legend(loc='upper right', fontsize='xx-small')
        plt.title("FID Score")
        plt.ylabel("FID Score")
        plt.xlabel("Epoch")
        plt.savefig("fid" + "_.png")




