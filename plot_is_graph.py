
import matplotlib.pyplot as plt
import os
path = r"C:\Users\eyalb\Desktop\Master\Courses\Generative_Models\HW\vae\VAE\gmpm\is_res\128_4_4"
res = []
for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
      epoch = int(name.split("_")[-2])
      with open(os.path.join(root, name), 'r') as file:
         data = file.read().replace('\n', '')
      ins = float(data.split(' ')[-1][8:-2])
      res.append((epoch,ins))
      label = "_".join(name.split("_")[:-3])

res_sort = sorted(res, key=lambda x: x[0])

tuples = zip(*res_sort)
epochs_vec, metrics_vec = [list(tuple) for tuple in tuples]

path = r"C:\Users\eyalb\Desktop\Master\Courses\Generative_Models\HW\vae\VAE\gmpm\is_res\128_2_4"
res = []
for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
      epoch = int(name.split("_")[-2])
      with open(os.path.join(root, name), 'r') as file:
         data = file.read().replace('\n', '')
      ins = float(data.split(' ')[-1][8:-2])
      res.append((epoch,ins))
      label2 = "_".join(name.split("_")[:-3])

res_sort = sorted(res, key=lambda x: x[0])

tuples = zip(*res_sort)
epochs_vec2, metrics_vec2 = [list(tuple) for tuple in tuples]



plt.plot(epochs_vec[2:], metrics_vec[2:],c = 'b', label=label, marker='o')
plt.plot(epochs_vec2[2:], metrics_vec2[2:],c = 'k', label=label2, marker='o')

plt.title("Inception Score")
plt.legend()
plt.ylabel("Inception Score")
plt.xlabel("Epoch")
plt.savefig("is" + "_.png")

v=0