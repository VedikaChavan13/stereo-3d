import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Load stereo images (grayscale)
left = cv2.imread("left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("right.png", cv2.IMREAD_GRAYSCALE)

if left is None or right is None:
    print("Images not found. Check filenames.")
    exit()

# 2. Create StereoSGBM matcher
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

# 3. Load KITTI calibration (000010.txt)
with open("000010.txt", "r") as f:
    lines = f.readlines()

# Projection matrices P0 and P1 (first two lines)
P0 = np.fromstring(lines[0].split(":", 1)[1], sep=" ").reshape(3, 4)
P1 = np.fromstring(lines[1].split(":", 1)[1], sep=" ").reshape(3, 4)

# Focal length (fx) and baseline B (meters)
f = P0[0, 0]                # ~707
Tx = P1[0, 3] / f           # camera x-translation (meters, negative)
B = -Tx                     # baseline is positive distance between cameras

print("Focal length f:", f)
print("Baseline B:", B)

# 4. Compute depth from disparity: depth = f * B / disparity
# Avoid division by zero or negative disparities
disp_valid = disp.copy()
disp_valid[disp_valid <= 0] = np.nan  # mark invalid

depth = f * B / disp_valid            # depth in meters

# 5. Simple visualization: clip depth for display (e.g., 0–80 m)

depth_clipped = np.clip(depth, 0, 80)

plt.figure(figsize=(10, 4))
plt.imshow(depth_clipped, cmap="inferno")
plt.title("Depth Map (meters, clipped)")
plt.axis("off")
plt.colorbar(label="meters")
plt.show()

np.save("depth_clipped.npy", depth_clipped)
print("Saved depth_clipped.npy")

