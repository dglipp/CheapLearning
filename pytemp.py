import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D as plt3d
import time

data = np.ones(shape=(256*256*256, 240))

def d3H(mat_input, n_bins = None):

    if n_bins is None:
        n_bins = int(mat_input.shape[0]/20)
    counts, bins= np.histogram(mat_input, bins = n_bins)
    counts = counts.T
    pos = np.empty(shape = n_bins)
    for i in range(n_bins):
        pos[i] = (bins[i] + bins[i+1])/2
    t_lags = np.linspace(0, mat_input.shape[1], mat_input.shape[1])
    
    a, b = np.meshgrid(t_lags, pos)
    
    return a, b, counts

stamp = time.time()
t, x, h = d3H(data)
print("data creation time: " + str(time.time()-stamp))
print(h.shape)
stamp = time.time()
fig = plt.figure()
ax = plt3d(fig)
ax.plot_surface(t, x, h, rstride=2, cstride=2)
print("plotting time: " + str(time.time()-stamp))
plt.show()







