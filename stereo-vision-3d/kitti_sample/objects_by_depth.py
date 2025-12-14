import numpy as np

FRAME_ID = "000003_11"  
pc_file = f"outputs/pc_{FRAME_ID}_f.txt"

points = np.loadtxt(pc_file)  # (N, 3)
print("Loaded points:", points.shape)

X = points[:, 0]
Y = points[:, 1]
Z = points[:, 2]


# Choose a depth band where cars/objects are likely, e.g. 10–30 m
z_min_obj, z_max_obj = 10.0, 30.0

mask = (Z >= z_min_obj) & (Z <= z_max_obj)
obj_points = points[mask]

print("Points in object band:", obj_points.shape)

np.savetxt(f"outputs/pc_{FRAME_ID}_objects_10_30m.txt", obj_points, fmt="%.4f")
print("Saved:", f"outputs/pc_{FRAME_ID}_objects_10_30m.txt")
