import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

# Load filtered point cloud
points = np.loadtxt("pointcloud_camera_xyz_filtered.txt")  # shape (N, 3)
print("Loaded filtered points:", points.shape)

if points.size == 0:
    print("No points to display. Check your filtering ranges.")
    exit()

# Optionally subsample for speed
N = points.shape[0]
max_points = 20000
if N > max_points:
    idx = np.random.choice(N, max_points, replace=False)
    points_vis = points[idx]
else:
    points_vis = points

X = points_vis[:, 0]  # right
Y = points_vis[:, 1]  # down
Z = points_vis[:, 2]  # forward

# 3D scatter plot (camera coordinates)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Use Z (forward distance) as color
p = ax.scatter(X, Z, -Y, c=Z, s=1, cmap='viridis')

ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_zlabel('Y (m, inverted)')
ax.set_title('KITTI Point Cloud (filtered, camera frame)')

fig.colorbar(p, label='Z (m)')
plt.tight_layout()
plt.show()
