import numpy as np
from sklearn.cluster import DBSCAN

FRAME_ID = "000003_11"
pc_file = f"outputs/pc_{FRAME_ID}_objects_10_30m.txt"  

points = np.loadtxt(pc_file)
print("Loaded object-slice points:", points.shape)

if points.size == 0:
    print("No points to cluster.")
    exit()

eps = 1.0
min_samples = 50
clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
labels = clustering.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Estimated number of clusters:", n_clusters)

out = np.concatenate([points, labels.reshape(-1, 1)], axis=1)
out_file = f"outputs/pc_{FRAME_ID}_objects_10_30m_clusters.txt"
np.savetxt(out_file, out, fmt="%.4f", header="X Y Z label")
print("Saved cluster file:", out_file)
