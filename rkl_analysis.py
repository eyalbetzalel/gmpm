import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d")

nll2prob = lambda a: np.exp(-1.0 * a)

def kld(p, q, Flag):

    # Original Calculation : (1.0 / len(p)) * sum((np.log(p[i] / q[i]) for i in range(len(p)))

    # Variance calculation using Jack-Knife Resampling method :

    # Step 1 - Calculate xi :

    if Flag == 1:
        log_vec = (1.0 / (len(p) - 1)) * (np.log(p) - np.log(q))
    elif Flag == 2:
        log_vec = (1.0 / (len(p) - 1)) * (np.log(p/q) + (q/p) - 1)
    else:
        log_vec = (1.0 / (len(p) - 1)) * (q/p)

    xi = [0] * len(p)

    for i in range(len(p)):
        xi[i] = np.sum(np.delete(log_vec, i))

    # Step 2 - Calculate Mean and VAR :

    mu = (1.0 / len(p)) * sum((xi[i]) for i in range(len(p)))
    var = ((len(p)-1) / len(p)) * sum(((xi[i]-mu) ** 2) for i in range(len(p)))

    return mu, var

path_imagegpt = r"C:\Users\eyalb\Desktop\Master\Courses\Generative_Models\HW\vae\VAE\gmpm\rkl_analysis\data\image-gpt"
path_pixelsnail = r"C:\Users\eyalb\Desktop\Master\Courses\Generative_Models\HW\vae\VAE\gmpm\rkl_analysis\data\pixelsnail"

data_imagegpt = []
data_pixelsnail = []


for root, dirs, files in os.walk(path_imagegpt, topdown=True):
    for name in files:
        epoch = int(name.split("_")[13])
        curr_data = np.load(os.path.join(root,name))
        data_imagegpt.append((epoch,curr_data))


for root, dirs, files in os.walk(path_pixelsnail, topdown=True):
    for name in files:
        epoch = int(name.split("_")[13])
        curr_data = pickle.load(open(os.path.join(root,name),"rb"))
        curr_data = [temp[1] for temp in curr_data]
        data_pixelsnail.append((epoch, curr_data))

data_pixelsnail = sorted(data_pixelsnail, key=lambda x: x[0])
data_imagegpt = sorted(data_imagegpt, key=lambda x: x[0])

import imageio
filenames = []

b = 0.8
# epsilon = 0.05
# gap = 1000

res_x = []
res_gap_b = []
res_gap_no_b = []
res1_err = []
res2_err = []

# while gap > 1.005 or gap < 0.9995:
p_mean = []
q_mean = []

for i in range(len(data_imagegpt)):
    epoch = data_imagegpt[i][0]
    # if epoch == 0:
    #     continue

    curr_pixelsnail = np.array(data_pixelsnail[i][1]) # q
    curr_imagegpt = data_imagegpt[i][1] # p


    q = nll2prob(curr_pixelsnail)
    p = nll2prob(curr_imagegpt)
    v=0



    q_tilda = np.exp(np.log(q) * b)
    gap1, bias_var1 = kld(q_tilda, p, 1)
    gap2, bias_var2 = kld(q, p, 1)
    res_x.append(epoch)
    res_gap_b.append(gap1)
    res_gap_no_b.append(gap2)
    res1_err.append(bias_var1)
    res2_err.append(bias_var2)

    p_mean.append(np.mean(p))
    q_mean.append(np.mean(q))

    # create file name and append it to a list
    # filename = f'{epoch}.png'
    # filenames.append(filename)


    # fig, ax = plt.subplots()
    # # bins = np.linspace(-5, 5, 25, endpoint=True)
    # plt.title("Likelihood Histogram - Epoch " + str(epoch))
    # ax.hist(p, color="blue", label="p (ImageGPT)", alpha=.9, bins=1000)
    # ax.hist(q, color="red", label="q (PixelSnail)", alpha=.6, bins=1000)
    # ax.set_xlim([0,0.1])
    # ax.set_ylim([0,1000])
    # plt.legend()
    # plt.savefig(filename)
    # plt.close()

# with imageio.get_writer(f"pdf_histogram_{timestamp}.gif", mode='I') as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)
#
# # Remove files
# for filename in set(filenames):
#     os.remove(filename)

# plt.axis = ([0, 45])
#
# plt.errorbar(res_x, res_gap_b, yerr=res_err, marker="o", label="bias =0.8")
# plt.errorbar(res_x, res_gap_no_b, yerr=res_err, marker="o", label="no bias)")
# plt.plot(np.ones(45), c="red", label="1", alpha=0.5)
#
# # plt.title("Reverse KL - PixelSnail VS. ImageGPT (Sample from PixelSnail)")
# plt.title("Bias Correction (Sample from PixelSnail)")
# plt.xlabel("Epoch")
# plt.ylabel("1 - Estimation")
# plt.legend()
# plt.show()

plt.axis = ([0, 45])

# plt.errorbar(res_x, res_gap_b, yerr=res1_err, marker="o", label="bias =0.8")
plt.errorbar(res_x, res_gap_no_b, yerr=res2_err, marker="o", label="KL_ext")
plt.plot(np.zeros(45), c="red", label="0", alpha=0.5)
plt.title("KL_ext (Sample from PixelSnail)")
plt.xlabel("Epoch")
plt.ylabel("RKL")
plt.legend()
plt.show()




v=0
