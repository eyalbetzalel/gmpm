import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# def js_divergence(p, q, mu1, sigma1, mu2, sigma2):
#     # TODO --> Bulid new KL function that can handle samples without pdf funtion.
#     m = [0.5 * (norm.pdf(p[i],mu1, sigma1) + norm.pdf(q[i],mu2, sigma2)) for i in range(len(p))]
#     return 0.5 * kl_divergence(norm.pdf(p[i],mu1, sigma1), m) + 0.5 * kl_divergence(norm.pdf(q[i],mu2, sigma2), m) for i in range(len(p)))

def kl_divergence(p, mu1, sigma1, mu2, sigma2):

    return (1.0 / len(p)) * sum(np.log(norm.pdf(p[i],mu1, sigma1)) - np.log(norm.pdf(p[i],mu2, sigma2)) for i in range(len(p)))

def kl_divergence_cf(mu1,sigma1,mu2,sigma2):
    return (np.log(sigma2/sigma1) + (np.power(sigma1,2) + (np.power((mu1-mu2),2)))/(2*np.power(sigma2,2)) - 0.5)

def kl_divergence_js(p, q):

    return (1.0 / len(p)) * sum(np.log(p[i]) - np.log(norm.pdf(q[i],mu2, sigma2)) for i in range(len(p)))

######## Sampling ########

mu1, sigma1 = 2, 1 # mean and standard deviation
p = np.random.normal(mu1, sigma1, 1000)

mu2, sigma2 = 1, 1 # mean and standard deviation

q = np.random.normal(mu2, sigma2, 1000)

######## Close Form - KL Divergnce ########

print("Close Form kl_div : " + str(kl_divergence_cf(mu1, sigma1, mu2, sigma2)))

######## MC Sampling - KL Divergnce ########

print("Monte Carlo Estimation kl_div : " + str(kl_divergence(p.tolist(), mu1, sigma1, mu2, sigma2)))

######## MC Sampling - JS Divergnce ########

# print("Monte Carlo Estimation js_div : " + str(js_divergence(p.tolist(), q.tolist(), mu1, sigma1, mu2, sigma2)))

######## Ploting ########

count, bins, ignored = plt.hist(p, 30, density=True)
plt.plot(bins, 1/(sigma1 * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu1)**2 / (2 * sigma1**2) ),
         linewidth=2, color='r')

count, bins, ignored = plt.hist(q, 30, density=True)
plt.plot(bins, 1/(sigma2 * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu2)**2 / (2 * sigma2**2) ),
         linewidth=2, color='r')
plt.show()