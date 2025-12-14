import cv2
import numpy as np

# 1. RECOMPUTE DISPARITY & DEPTH 
left = cv2.imread("left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("right.png", cv2.IMREAD_GRAYSCALE)

if left is None or right is None:
    print("Images not found. Check filenames.")
    exit()

# Stereo matcher
min_disp = 0
num_disp = 64  # must be divisible by 16
block_size = 5

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1
)

disp = stereo.compute(left, right).astype(np.float32) / 16.0

# Load calibration
with open("000010.txt", "r") as f:
    lines = f.readlines()

P0 = np.fromstring(lines[0].split(":", 1)[1], sep=" ").reshape(3, 4)
P1 = np.fromstring(lines[1].split(":", 1)[1], sep=" ").reshape(3, 4)

f = P0[0, 0]
Tx = P1[0, 3] / f
B = -Tx

print("Focal length f:", f)
print("Baseline B:", B)

# Depth = f * B / disparity
disp_valid = disp.copy()
disp_valid[disp_valid <= 0] = np.nan
depth = f * B / disp_valid  # meters

# 2. BACK-PROJECT TO 3D POINT CLOUD 
fx = P0[0, 0]
fy = P0[1, 1]
cx = P0[0, 2]
cy = P0[1, 2]

h, w = left.shape
u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

z = depth.flatten()
u = u_coords.flatten()
v = v_coords.flatten()

valid = ~np.isnan(z)
z = z[valid]
u = u[valid]
v = v[valid]

X = (u - cx) * z / fx
Y = (v - cy) * z / fy
Z = z

points_3d = np.vstack((X, Y, Z)).T

print("3D point cloud shape:", points_3d.shape)
print("First 5 points:\n", points_3d[:5])

np.savetxt("pointcloud_camera_xyz.txt", points_3d, fmt="%.4f")
print("Saved pointcloud_camera_xyz.txt")
