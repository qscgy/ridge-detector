import numpy as np
import matplotlib.pyplot as plt

fpath = '/playpen/Datasets/geodepth2/041/NFPS/images/041_037/041_037_nr_pred.npy'
normals = np.load(fpath)

u = normals[::10,::10,0]
v = normals[::10,::10,1]
w = normals[::10,::10,2]

x, y, z = np.meshgrid(np.linspace(0, 1, u.shape[0]), np.linspace(0, 1, u.shape[1]), np.linspace(0,1,1))
ax = plt.figure().add_subplot(projection='3d')
ax.quiver(x, y, z, normals[::10,::10,0], normals[::10,::10,1], normals[::10,::10,2], length=0.01)
plt.show()