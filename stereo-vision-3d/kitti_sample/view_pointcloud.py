import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

# Load point cloud file
points = np.loadtxt("pointcloud_camera_xyz.txt")  # shape (N, 3)

print("Loaded points:", points.shape)

# Optionally subsample for faster plotting
N = points.shape[0]
max_points = 20000
if N > max_points:
    idx = np.random.choice(N, max_points, replace=False)
    points_vis = points[idx]
else:
    points_vis = points

X = points_vis[:, 0]
Y = points_vis[:, 1]
Z = points_vis[:, 2]

# Basic 3D scatter plot (camera coords: X right, Y down, Z forward)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(X, Z, -Y, c=Z, s=1, cmap='viridis')  # use Z as color

ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_zlabel('Y (m, inverted)')
ax.set_title('KITTI Point Cloud (camera frame)')
fig.colorbar(p, label='Z (m)')
plt.show()
