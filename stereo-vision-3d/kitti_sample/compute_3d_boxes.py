import numpy as np

FRAME_ID = "000003_11"
cluster_file = f"outputs/pc_{FRAME_ID}_objects_10_30m_clusters.txt"

data = np.loadtxt(cluster_file, comments="#")
points = data[:, :3]
labels = data[:, 3].astype(int)

unique_labels = sorted(l for l in set(labels) if l != -1)
print("Clusters (excluding noise):", unique_labels)

boxes = []
for lab in unique_labels:
    pts = points[labels == lab]
    if pts.shape[0] < 50:
        continue
    x_min, y_min, z_min = pts.min(axis=0)
    x_max, y_max, z_max = pts.max(axis=0)
    boxes.append([lab, x_min, y_min, z_min, x_max, y_max, z_max])

boxes = np.array(boxes)
print("Computed boxes:", boxes.shape)

out_path = f"outputs/boxes3d_{FRAME_ID}.txt"
np.savetxt(out_path, boxes,
           fmt="%.4f",
           header="label x_min y_min z_min x_max y_max z_max")
print("Saved 3D boxes to", out_path)
