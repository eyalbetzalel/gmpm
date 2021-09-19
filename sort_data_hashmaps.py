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

#############################################################################################################

# import :

import os
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import ot

# args :

parser = argparse.ArgumentParser()

parser.add_argument(
    "--imagegpt_res",
    type=str,
    help="Path to ImageGPT results",
    default=r"C:\Users\eyalb\Desktop\Master\Courses\Generative_Models\HW\vae\VAE\gmpm\imageGPT_Evaluation_Results_NLL.p",
)

parser.add_argument(
    "--pixel_snail_data",
    type=str,
    help="Path to PixelSnail data",
    default=r"C:\Users\eyalb\Desktop\Master\Courses\Generative_Models\HW\vae\VAE\gmpm\test_imagegpt.h5",
)

# r"C:\Users\eyalb\Desktop\Master\Courses\Generative_Models\HW\vae\VAE\gmpm\dataset_only_from_pixelsnail.p"
# r"C:\Users\eyalb\Desktop\Master\Courses\Generative_Models\HW\vae\VAE\gmpm\train_imagegpt.h5"
parser.add_argument(
    "--pixelsnail_res",
    type=str,
    help="Path to PixelSnail results",
    default=r"C:\Users\eyalb\Desktop\log",
)
mode = "test"
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

    elif path.endswith('.h5'):

        with h5py.File(path, "r") as f:
            a_group_key = list(f.keys())[0]
            # Get the data
            model_table = list(f[a_group_key])

    return model_table

def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)

p = []  # p is likelihood of ImageGPT
q = []  # q is likelihood of PixelSnail

def kld(p, q, revFlag):

    # Original Calculation : (1.0 / len(p)) * sum((np.log(p[i] / q[i]) for i in range(len(p)))

    # Variance calculation using Jack-Knife Resampling method :

    # Step 1 - Calculate xi :

    p_np = np.asarray(p)
    q_np = np.asarray(q)
    if revFlag:
        # Since we sample from p and want to estimate the reverse KL over q , we need to multiply by factor due to
        # importance sampling.
        log_vec = (1.0 / (len(p) - 1)) * np.log(q_np / p_np) * (q_np/p_np)
    else:
        log_vec = (1.0 / (len(p) - 1)) * np.log(p_np / q_np)

    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(log_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p)-1) / len(p)) * sum(((xi[i]-mu) ** 2) for i in range(len(p)))

    return mu, var

def tvd(p, q):

    # Total Variation Distance Calculation
    # Original Calculation : (1.0 / len(p)) * sum(0.5 * np.abs(p[i]/q[i] - 1) for i in range(len(p)))
    # Variance calculation using Jack-Knife Resampling method.

    # Step 1 - Calculate xi :

    p_np = np.asarray(p)
    q_np = np.asarray(q)
    tvd_vec = (1.0 / (len(p) - 1)) * (0.5 * np.abs((p_np/q_np) - 1))
    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(tvd_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p) - 1) / len(p)) * sum(((xi[i] - mu) ** 2) for i in range(len(p)))

    return mu, var

def jsd(p, q):

    # Jensen-Shannon Divergence Calculation :
    # Original Calculation (1) : m = [0.5 * (p[i] + q[i]) for i in range(len(p))]
    # Original Calculation (2) : return 0.5 * kld(p, m) + 0.5 * kld(q, m)

    p_np = np.asarray(p)
    q_np = np.asarray(q)
    m_np = 0.5 * (p_np + q_np)
    log_vec = (1.0 / (len(p) - 1)) * 0.5 * (np.log(p_np / m_np) + np.log(q_np / m_np) * (q_np/p_np))
    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(log_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p)-1) / len(p)) * sum(((xi[i]-mu) ** 2) for i in range(len(p)))

    return mu, var

def otd(p, q):

    # Unbalanced Optimal transport using a Kullback-Leibler relaxation.

    p_np = np.asarray(p)
    q_np = np.asarray(q)
    n = len(p)

    # loss matrix

    M = np.identity(n-1)
    xi = [0] * len(p)

    epsilon = 0.1  # entropy parameter
    alpha = 1.  # Unbalanced KL relaxation parameter

    for i in range(len(p)):

        curr_p = np.delete(p_np, i)
        curr_q = np.delete(q_np, i)
        xi[i] = (1.0 / (len(p) - 1)) * ot.sinkhorn_unbalanced2(curr_p, curr_q, M, 1, 1, verbose=True)


    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p)-1) / len(p)) * sum(((xi[i]-mu) ** 2) for i in range(len(p)))

    return mu, var

def chi2(p,q):
    # chi square divergence
    # Estimate is 1/m \sum_i dQn(zi) / dP(zi) - 1 with zi ~ Qn.
    # squared Hellinger distance
    # Estimate is 2 - 2 / m \sum_i exp(0.5 log (dP(zi) / dQn(zi))), zi ~ Qn.
    p_np = np.asarray(p)
    q_np = np.asarray(q)
    log_vec = (1.0 / (len(p) - 1)) * ((p_np / q_np)-1)
    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(log_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p) - 1) / len(p)) * sum(((xi[i] - mu) ** 2) for i in range(len(p)))

    return mu, var

def hsq(p,q):
    # squared Hellinger distance
    # Estimate is 2 - 2 / m \sum_i exp(0.5 log (dP(zi) / dQn(zi))), zi ~ Qn.
    p_np = np.asarray(p)
    q_np = np.asarray(q)
    log_vec = 2 * (1 - (1.0 / (len(p) - 1)) * np.exp(0.5 * np.log(q_np / p_np)))
    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(log_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p) - 1) / len(p)) * sum(((xi[i] - mu) ** 2) for i in range(len(p)))

    return mu, var


nll2prob = lambda a: np.exp(-1 * a)/1024

def plot_graph(title,epochs, metrics, labels,colors):



    plt.figure()

    for i in range(len(epochs)):
        if labels[i].split("_")[0] != "ep":
            continue
        zipped_lists = zip(epochs[i], metrics[i])
        sorted_pairs = sorted(zipped_lists)
        tuples = zip(*sorted_pairs)
        epochs_vec, metrics_vec = [list(tuple) for tuple in tuples]
        mu = [x[0] for x in metrics_vec]
        var = [x[1] for x in metrics_vec]
        plt.errorbar(epochs_vec, mu, yerr=var, c=colors(2*i), label=labels[i], marker='o')

        rows = [epochs_vec, mu, var]
        np.savetxt(title + "_" + labels[i] + "_" + mode + ".csv",
                   rows,
                   delimiter=", ",
                   fmt='% s')

    plt.title(title + " Score - " + mode + " Set")
    plt.legend()
    plt.ylabel(title)
    plt.xlabel("Epoch")
    plt.savefig(title + "_.png")

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, 2*n)

# Import datasets:

imagegpt_dataset = load_file(args.imagegpt_res)
pixelsnail_data = load_file(args.pixel_snail_data)

ot_d = []
kl = []
js = []
tot_var = []
rev_kl = []
chi = []
hs = []
label = []
epochss = []


for root, dirs, files in os.walk(args.pixelsnail_res):

    curr_str = root.split("\\")
    curr_str = curr_str[-1]
    curr_kl = []
    curr_js = []
    curr_var_dist = []
    curr_rev_kl = []
    curr_ot_d = []
    curr_chi = []
    curr_hs = []
    epochs = []


    for filestr in files:
        if filestr.endswith("_" + mode + "_eval.p"):

            #train_eval
            print(" Working on : %s" % filestr)

            # Import current model evaluation results:

            print(" == Load data ==")

            pixelsnail_res = load_file(os.path.join(root, filestr))

            # Join data to NLL results :
            if pixelsnail_data[1].shape[0] == 1024:

                for i in range(len(pixelsnail_data)):
                    pixelsnail_data[i] = np.expand_dims(pixelsnail_data[i],0)

            model_pixelsnail = []
            model_pixelsnail = [(pixelsnail_data[i], pixelsnail_res[i]) for i in range(len(pixelsnail_data))]

            # Sort ImageGPT and PixelSnail results in the same order --> Create hashmap :

            d = {}

            print(" == Creating hashmaps ==")

            for i in tqdm(range(len(model_pixelsnail))):
                key = tuple(map(tuple, model_pixelsnail[i][0].astype(np.int)))
                add_element(d, key, i)

            sorted_list = []
            [sorted_list.append(0) for _ in range(len(model_pixelsnail))]

            # Check if lists are coordinated and sort ImageGPT list:

            print(" == Sorting ImageGPT ==")
            for i in tqdm(range(len(imagegpt_dataset))):
                curr = np.expand_dims(imagegpt_dataset[i][0], axis=0)
                key = tuple(map(tuple, curr.astype(int)))
                value = d.get(key)

                if value == None:
                    continue

                value = value[0]
                sorted_list[value] = (imagegpt_dataset[i][0], imagegpt_dataset[i][1], model_pixelsnail[value][1])

            # Calc f - divergence  :

            # samples took from ImageGPT and the likelihood estmation is preform from imageGPT and PixelSnail :

            p = [] # p is likelihood of ImageGPT
            q = [] # q is likelihood of PixelSnail

            print(" == Calculating f div ==")

            for i in tqdm(range(len(sorted_list))):
                if sorted_list[i] == 0:
                    continue
                p.append(nll2prob(sorted_list[i][1]))
                q.append(nll2prob(sorted_list[i][2]))


            # ot_d_res = otd(p, q)
            kld_mu, kld_var = kld(p, q, False)
            rev_kld_mu, rev_kld_var = kld(p, q, True)
            tvd_mu, tvd_var = tvd(p, q)
            jsd_mu, jsd_var = jsd(p, q)
            chi_mu, chi_var = chi2(p, q)
            hs_mu, hs_var = hsq(p, q)
            curr_run_str = filestr[:-2]

            kld_res = (kld_mu, kld_var)
            jsd_res = (jsd_mu, jsd_var)
            tvd_res = (tvd_mu, tvd_var)
            rev_kld_res = (rev_kld_mu, rev_kld_var)
            chi_res = (chi_mu, chi_var)
            hs_res = (hs_mu, hs_var)

            # Saving results to text file :

            # file_name = curr_run_str + ".txt"
            # file_name = os.path.join(root, file_name)
            # file = open(file_name,"a")
            #
            # curr_run_str = curr_run_str
            # kl_d_str = "kl_d = " + str(kld_res)
            # rev_kl_d_str = "rev_kl_d = " + str(rev_kld_res)
            # tot_var_dist_str = "tot_var_dist = " + str(tvd_res)
            # jsd_str = "js_divergence = " + str(jsd_res)
            #
            # file.write('%r\n%r\n%r\n%r\n%r\n' % (curr_run_str, kl_d_str, rev_kl_d_str, tot_var_dist_str, jsd_str))
            # file.close()

            curr_kl.append(kld_res)
            curr_js.append(jsd_res)
            curr_var_dist.append(tvd_res)
            curr_rev_kl.append(rev_kld_res)
            curr_chi.append(chi_res)
            curr_hs.append(hs_res)

            epoch = int(filestr.split("_")[-3])
            epochs.append(epoch)

    kl.append(curr_kl)
    js.append(curr_js)
    tot_var.append(curr_var_dist)
    rev_kl.append(curr_rev_kl)
    chi.append(curr_chi)
    hs.append(curr_hs)
    epochss.append(epochs)
    label.append(curr_str)

cmap = plt.cm.get_cmap('hsv', 10)

plot_graph('KL',epochss,kl,label,cmap)
plot_graph('JS',epochss,js,label,cmap)
plot_graph('Total Variation Distance',epochss,tot_var,label,cmap)
plot_graph('Reverse KL',epochss,rev_kl,label,cmap)
plot_graph('Chi2',epochss,chi,label,cmap)
plot_graph('HS',epochss,hs,label,cmap)


