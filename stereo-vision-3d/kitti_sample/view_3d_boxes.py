import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

FRAME_ID = "000010_10"
pc_file = f"outputs/pc_{FRAME_ID}_objects_10_30m_clusters.txt"
box_file = f"outputs/boxes3d_{FRAME_ID}.txt"

# Load clustered points (X, Y, Z, label)
data = np.loadtxt(pc_file, comments="#")
points_all = data[:, :3]
labels_all = data[:, 3].astype(int)

# Keep only non-noise points once
mask = labels_all != -1
points = points_all[mask]
labels = labels_all[mask]

# Load boxes
boxes = np.loadtxt(box_file, comments="#")
print("Loaded points:", points.shape)
print("Loaded boxes:", boxes.shape)

X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot clustered points
scatter = ax.scatter(X, Z, -Y, c=labels, s=2, cmap='tab20')

def draw_box(ax, x_min, y_min, z_min, x_max, y_max, z_max, color='k'):
    xs = [x_min, x_max]
    ys = [y_min, y_max]
    zs = [z_min, z_max]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])

    edges = [
        (0,1), (0,2), (0,4),
        (3,1), (3,2), (3,7),
        (5,1), (5,4), (5,7),
        (6,2), (6,4), (6,7)
    ]
    for i, j in edges:
        x = [corners[i, 0], corners[j, 0]]
        y = [corners[i, 1], corners[j, 1]]
        z = [corners[i, 2], corners[j, 2]]
        ax.plot(x, z, -np.array(y), color=color, linewidth=1.5)

# Draw boxes
for row in boxes:
    lab, x_min, y_min, z_min, x_max, y_max, z_max = row
    draw_box(ax, x_min, y_min, z_min, x_max, y_max, z_max, color='k')

ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_zlabel("Y (m, inverted)")
ax.set_title("3D clusters + bounding boxes (10–30 m)")

plt.tight_layout()
plt.show()
