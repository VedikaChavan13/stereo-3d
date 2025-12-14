import numpy as np

# Load raw point cloud
points = np.loadtxt("pointcloud_camera_xyz.txt")  # (N, 3)
print("Original points:", points.shape)

X = points[:, 0]
Y = points[:, 1]
Z = points[:, 2]

# Keep only reasonable ranges 
z_min, z_max = 1.0, 80.0      # distance in front of camera (meters)
x_min, x_max = -20.0, 20.0    # left/right span
y_min, y_max = -5.0, 5.0      # vertical span

mask = (
    (Z >= z_min) & (Z <= z_max) &
    (X >= x_min) & (X <= x_max) &
    (Y >= y_min) & (Y <= y_max)
)

points_f = points[mask]
print("Filtered points:", points_f.shape)

np.savetxt("pointcloud_camera_xyz_filtered.txt", points_f, fmt="%.4f")
print("Saved pointcloud_camera_xyz_filtered.txt")
