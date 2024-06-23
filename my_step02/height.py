import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

path = os.path.join(os.path.dirname(__file__), 'height.txt')
xs = np.loadtxt(path)
print(xs.shape)

mu = np.mean(xs)
sigma = np.std(xs)

x = np.linspace(150, 190, 1000)
pdf = norm.pdf(x, loc=mu, scale=sigma)

plt.hist(xs, bins='auto', density=True)
plt.plot(x, pdf)
plt.xlabel('Height (cm)')
plt.ylabel('Probability density')
plt.show()