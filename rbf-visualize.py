# 3D point cloud interpolation
# we'll sample points from a spehere 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
N = 250
phi = 2 * np.pi * np.random.rand(N)
costheta = 2*np.random.rand(N) - 1
theta = np.arccos(costheta)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
points = np.vstack([x, y, z]).T # N rows, 3 cols

#Normals 
normals = points.copy()

delta = 0.05
points_in = points - delta * normals
points_out = points + delta * normals

# combine all poitns
Q = np.vstack([
    points,              # original surface points 
    points_in,           # generated inside-points
    points_out           # generated outside-points
])
# Q has 3N rows, 3 cols

f = np.concatenate([
    np.zeros(N),          # original surface points
    -delta*np.ones(N),    # generated inside points
    +delta*np.ones(N)     # generated outside points
])
# f has 3N rows, 1 col

# Gaussian RBF
def gaussian_rbf(r, epsilon=2.0):
    return np.exp(-(epsilon * r)**2)

# Pairwise distance
def pairwise_dist(A, B):
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)

# build block matrix equation
epsilon = 2.0
Phi = gaussian_rbf(pairwise_dist(Q, points), epsilon)
# Phi has 3N rows and N cols

# Polynomial terms
P = np.hstack([np.ones((len(Q),1)), Q])
# P has 3N rows and 4 cols

# A = np.block([
#     [Phi, P],
#     [P.T, np.zeros((4,4))]
# ])

A = np.block([
    [Phi, P]
])

# b = np.concatenate([f,np.zeros(4)])
b = f.copy()

# solve for \vec{omega} and \vec{a}
w_a = np.linalg.lstsq(A,b,rcond=None)[0]
w = w_a[:-4]
a = w_a[-4:]

# Evaluate on a grid
grid_x,grid_y,grid_z = [np.linspace(-1.5,1.5,40)]*3
X,Y,Z = np.meshgrid(grid_x,grid_y,grid_z)
grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

Phi_eval = gaussian_rbf(pairwise_dist(grid, points), epsilon)
P_eval = np.hstack([np.ones((len(grid),1)), grid])
f_eval = Phi_eval @ w + P_eval @ a
f_eval = f_eval.reshape(X.shape)

# Visualize implicit surface (f=0)
from skimage import measure
verts, faces, normals, values = measure.marching_cubes(f_eval, level=0, spacing=(grid_x[1]-grid_x[0],)*3)

# Plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:,0]-1.5, verts[:,1]-1.5, faces, verts[:,2]-1.5,
                color='lightblue', alpha=0.6, linewidth=10)
ax.scatter(points[:,0], points[:,1], points[:,2], color='r', s=10)
ax.set_title("RBF Surface Reconstruction from Point Cloud")
plt.show()
