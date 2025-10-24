import numpy as np
from matplotlib import pyplot as plt

# Sample problem, Symmetric Positive Definite Matrix
A = np.array([[3,2],[2,6]])
b = np.array([2,-8])
c = 0

def quadratic_form(x, y, A,b,c):
    # Create the vector [x, y] and calculate the quadratic form
    vec = np.array([x, y])
    return 0.5*vec.T @ A @ vec + b.T @ vec + c

def gradient_quadratic_form(x,y,A):
    vec = np.array([x,y])
    return A @ vec


# generate a grid of points like [[x1,y1],[x2,y2],[x3,y3]] and evalue f at each point
ext,divs = 5,5
x = np.linspace(-ext,ext,divs)
y = np.linspace(-ext,ext,divs)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = quadratic_form(X[i, j], Y[i, j], A,b,c)

# Gradients

xp, yp = 3, 4
zp = quadratic_form(xp, yp, A,b,c)
gp = gradient_quadratic_form(xp, yp, A)
point = np.array([xp,yp,zp])
dfdx, dfdy = gp
# surface is z = f(x, y), the 3D normal is (df/dx, df/dy, -1)
normal = np.array([gp[0],gp[1],-1])


# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate d and we're set
d = -point.dot(normal)
# z = ax + by + d because c = -1
z_tangent = (normal[0] * X + normal[1] * Y + d)



# normal surface
xx = np.linspace(xp - 0.5*dfdx, xp + 0.5*dfdx,10)
yy = np.linspace(yp - 0.5*dfdy, yp + 0.5*dfdy,10)
XX, YY = np.meshgrid(xx,yy)
z_normal = (dfdx * XX + dfdy * YY) + d

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis',alpha=0.3)
# ax.plot_surface(X,Y, z_tangent,alpha=0.3)
# ax.plot_surface(XX,YY, z_normal,alpha=0.5)

dZdx,dZdy = np.gradient(Z,x,y)
ax.quiver(X, Y, Z, X + X*dZdx, Y+Y*dZdy, 1, length=0.005, color='r')

# Plot the point (1, 2, f(1,2))
ax.scatter(xp, yp, zp, color='black', s=50, label='Point (1,2) on surface')


# Add labels for axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')

ax.set_title('$z(x,y) = \\vec{x}^T \\mathbf{A} \\vec{x} + \\vec{b}^T \\vec{x} + c$')

plt.show()



