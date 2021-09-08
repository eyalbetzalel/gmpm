
import matplotlib.pyplot as plt
import os
import numpy as np

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, 2*n)


color = get_cmap(3)
path = r"C:\Users\eyalb\Desktop\Master\Courses\Generative_Models\HW\vae\VAE\gmpm\res_test"
res = []
i=0
for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
       if name.endswith('.csv'):
          l = np.loadtxt(open(os.path.join(root, name), "rb"), delimiter=",")
          label = "_".join(name.split("_")[:-3])
          plt.plot(l[0,2:], l[1,2:],c = color(i), label=label, marker='o')
          i+=1

plt.title("KL Divergence")
plt.legend()
plt.ylabel("KL")
plt.xlabel("Epoch")
plt.savefig("test" + "_.png")

v=0