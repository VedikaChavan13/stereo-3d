import cv2
import numpy as np

FRAME_ID = "000003_11"

left_img_path = "/Users/vedikachavan/Downloads/data_stereo_flow/testing/image_0/" + FRAME_ID + ".png"
img = cv2.imread(left_img_path)
if img is None:
    print("Could not load image:", left_img_path)
    exit()

h, w, _ = img.shape

with open("000010.txt", "r") as f:   
    lines = f.readlines()
P0 = np.fromstring(lines[0].split(":", 1)[1], sep=" ").reshape(3, 4)

fx = P0[0, 0]
fy = P0[1, 1]
cx = P0[0, 2]
cy = P0[1, 2]

boxes = np.loadtxt(f"outputs/boxes3d_{FRAME_ID}.txt", comments="#")
print("Loaded boxes:", boxes.shape)

for row in boxes:
    label, x_min, y_min, z_min, x_max, y_max, z_max = row
    xs = [x_min, x_max]
    ys = [y_min, y_max]
    zs = [z_min, z_max]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])

    X, Y, Z = corners[:, 0], corners[:, 1], corners[:, 2]
    Z[Z <= 0] = 1e-6

    u = (fx * X / Z) + cx
    v = (fy * Y / Z) + cy

    u_min, u_max = int(np.clip(u.min(), 0, w - 1)), int(np.clip(u.max(), 0, w - 1))
    v_min, v_max = int(np.clip(v.min(), 0, h - 1)), int(np.clip(v.max(), 0, h - 1))
    if u_max <= u_min or v_max <= v_min:
        continue

    color = (0, 255, 0)
    cv2.rectangle(img, (u_min, v_min), (u_max, v_max), color, 2)
    cv2.putText(img, f"{int(label)}", (u_min, max(v_min - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

out_path = f"outputs/boxes2d_{FRAME_ID}.png"
cv2.imwrite(out_path, img)
print("Saved 2D boxes on image to", out_path)
