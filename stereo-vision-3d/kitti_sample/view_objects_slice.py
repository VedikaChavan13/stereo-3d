import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

FRAME_ID = "000010_10"
pc_file = f"outputs/pc_{FRAME_ID}_objects_10_30m.txt"

points = np.loadtxt(pc_file)
print("Loaded object-slice points:", points.shape)

if points.size == 0:
    print("No points in this depth band.")
    exit()

# Optional subsample
N = points.shape[0]
max_points = 30000
if N > max_points:
    idx = np.random.choice(N, max_points, replace=False)
    points_vis = points[idx]
else:
    points_vis = points

X = points_vis[:, 0]
Y = points_vis[:, 1]
Z = points_vis[:, 2]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X, Z, -Y, c=Z, s=1, cmap='viridis')

ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_zlabel('Y (m, inverted)')
ax.set_title('Objects depth slice (10–30 m)')

fig.colorbar(p, label='Z (m)')
plt.tight_layout()
plt.show()
