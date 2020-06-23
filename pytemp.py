import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D as plt3d
import time

#wrapper for np.histogram
def utilhist(data, bins, vrange):
    counts, _ = np.histogram(data, bins = bins, range=vrange)
    return counts

#shape[0] (row) needs to be time 
#the function automatically flattens remaining dimensions
#if rang isn't setted uses whole range
def d3H(mat_input, n_bins = 500, rang = None):
    if rang is None:
        rang = (np.min(mat_input)-1, np.max(mat_input) +1) 
    counts = (np.apply_along_axis(utilhist, 1, data, bins = n_bins, vrange=rang)).T
    dif = (rang[1] - rang[0])/n_bins/2
    pos = np.linspace(rang[0]+ dif, rang[1]- dif, n_bins)
    t_lags = np.linspace(0, mat_input.shape[0], mat_input.shape[0])
    a, b = np.meshgrid(t_lags, pos)
    return a, b, counts

#input data
data = np.random.randn(400, 256*256) + 100

stamp = time.time()
#function call
t, x, h = d3H(data, n_bins = 100)

print("data creation time: " + str(time.time()-stamp))

stamp = time.time()
#plot section
fig = plt.figure()
ax = plt3d(fig)
ax.plot_surface(t, x, h, rstride=3, cstride=3)
ax.set_title("Plot title")
ax.set_xlabel("t")
ax.set_ylabel("variable (emission)")
ax.set_zlabel("Counts (density)")
fig.savefig('./buba.pdf') #change dir address on server
print("plotting time: " + str(time.time()-stamp))
plt.show() #comment on server