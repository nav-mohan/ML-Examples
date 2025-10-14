import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# --- Generate a noisy surface: z = sin(x)*cos(y) ---
np.random.seed(2)
n = 500
x = np.random.uniform(-2*np.pi, 2*np.pi, n)
y = np.random.uniform(-2*np.pi, 2*np.pi, n)
z_true = np.sin(x) * np.cos(y)
z_noisy = z_true + 0.2*np.random.randn(n)
points = np.vstack([x, y, z_noisy]).T

# --- Weight function ---
def gaussian_weight(r, h):
    return np.exp(-(r**2)/(h**2))

# --- Pick a query point ---
i0 = 100  # index of point to visualize
p0 = points[i0]

# --- Neighborhood and weights ---
k = 40
h = 1.0
tree = cKDTree(points)
dists, idx = tree.query(p0, k=k)
neighbors = points[idx]
w = gaussian_weight(dists, h)
W = np.diag(w)

# --- Weighted centroid and PCA ---
p_bar = np.sum(w[:,None]*neighbors, axis=0)/np.sum(w)
X = neighbors - p_bar
C = X.T @ W @ X
eigvals, eigvecs = np.linalg.eigh(C)
n_vec = eigvecs[:,0]    # normal
u_axis, v_axis = eigvecs[:,1], eigvecs[:,2]

# --- Project neighbors to (u,v) frame ---
U = X @ u_axis
V = X @ v_axis
Z = X @ n_vec

# --- Fit quadratic polynomial z(u,v) ---
A = np.vstack([np.ones_like(U), U, V, U**2, U*V, V**2]).T
ATA = A.T @ W @ A
ATZ = A.T @ W @ Z
a = np.linalg.solve(ATA, ATZ)

# --- Evaluate on grid in local coords ---
u_grid = np.linspace(U.min(), U.max(), 20)
v_grid = np.linspace(V.min(), V.max(), 20)
Umesh, Vmesh = np.meshgrid(u_grid, v_grid)
Zmesh = (a[0] + a[1]*Umesh + a[2]*Vmesh + 
         a[3]*Umesh**2 + a[4]*Umesh*Vmesh + a[5]*Vmesh**2)

# --- Convert mesh back to 3D space ---
surface_pts = (p_bar 
               + Umesh[...,None]*u_axis
               + Vmesh[...,None]*v_axis
               + Zmesh[...,None]*n_vec)

# --- MLS projection of p0 ---
z_fit = a[0]
p_proj = p_bar + z_fit*n_vec

# --- Plot everything ---
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# neighbors
ax.scatter(neighbors[:,0], neighbors[:,1], neighbors[:,2], color='gray', s=15, label='neighbors')

# tangent plane (flat)
plane_pts = (p_bar 
             + Umesh[...,None]*u_axis 
             + Vmesh[...,None]*v_axis)
ax.plot_surface(plane_pts[...,0], plane_pts[...,1], plane_pts[...,2],
                alpha=0.3, color='blue', label='tangent plane')

# fitted MLS patch
ax.plot_surface(surface_pts[...,0], surface_pts[...,1], surface_pts[...,2],
                alpha=0.6, color='red', label='MLS fit')

# highlight point and projection
ax.scatter(*p0, color='k', s=60, label='query point')
ax.scatter(*p_proj, color='red', s=60, marker='*', label='MLS projection')

ax.set_title("Local MLS Surface Fit")
ax.legend()
plt.show()
