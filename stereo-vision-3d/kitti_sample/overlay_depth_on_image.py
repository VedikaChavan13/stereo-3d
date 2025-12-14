import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load original left image (color)
left_color = cv2.imread("left.png")
left_rgb = cv2.cvtColor(left_color, cv2.COLOR_BGR2RGB)

# Load clipped depth
depth_clipped = np.load("depth_clipped.npy")  # shape HxW

# Use only reasonable range, e.g. 5–40 m, and ignore NaNs
depth_for_vis = np.clip(depth_clipped, 5, 40)
valid = ~np.isnan(depth_for_vis)
valid_vals = depth_for_vis[valid]

# If there are valid depths, normalize them
if valid_vals.size > 0:
    d_min, d_max = valid_vals.min(), valid_vals.max()
    # Avoid zero range
    if d_max == d_min:
        d_max = d_min + 1.0
    depth_norm = (depth_for_vis - d_min) / (d_max - d_min)
    depth_norm[~valid] = 0
    depth_norm = (depth_norm * 255).astype(np.uint8)
else:
    depth_norm = np.zeros_like(depth_for_vis, dtype=np.uint8)

# Colormap
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

# Blend original image and depth colormap
alpha = 0.6  # change between 0 (only image) and 1 (only depth)
overlay = cv2.addWeighted(left_rgb, 1 - alpha, depth_color, alpha, 0)

# Show all three
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Left Image")
plt.imshow(left_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Depth Colormap")
plt.imshow(depth_color)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay (Image + Depth)")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()

# After loading depth_clipped
depth_clipped = np.load("depth_clipped.npy")

# Focus on nearer range to enhance contrast, e.g. 5–60 m
depth_for_vis = np.clip(depth_clipped, 5, 60)

depth_norm = cv2.normalize(
    depth_for_vis, None, alpha=0, beta=255,
    norm_type=cv2.NORM_MINMAX
).astype(np.uint8)

