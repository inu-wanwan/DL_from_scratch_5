import os 
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

N = 10000
D = 2
xs = np.random.rand(N, D)

mu = np.sum(xs, axis=0) / N

cov = 0

for n in range(N):
    x = xs[n]
    z = x - mu
    z = z[:, np.newaxis]
    cov += z @ z.T

cov /= N

print(mu)
print(cov)