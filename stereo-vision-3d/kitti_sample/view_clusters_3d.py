import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt

FRAME_ID = "000010_10"
pc_file = f"outputs/pc_{FRAME_ID}_objects_10_30m_clusters.txt"

data = np.loadtxt(pc_file, comments="#")
points = data[:, :3]
labels = data[:, 3].astype(int)

print("Points:", points.shape)

# Remove noise label -1 for coloring
mask = labels != -1
points = points[mask]
labels = labels[mask]

if points.size == 0:
    print("No clustered points to show.")
    exit()

X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X, Z, -Y,
    c=labels,
    s=3,
    cmap='tab20'
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_zlabel("Y (m, inverted)")
ax.set_title("3D clusters in 10–30 m depth band")

plt.tight_layout()
plt.show()
