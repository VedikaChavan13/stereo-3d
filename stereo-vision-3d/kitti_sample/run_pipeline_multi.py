import cv2
import numpy as np
from pathlib import Path

# CONFIG 
KITTI_ROOT = Path("/Users/vedikachavan/Downloads/data_stereo_flow/testing")
IMAGE0_DIR = KITTI_ROOT / "image_0"     # left
IMAGE1_DIR = KITTI_ROOT / "image_1"     # right
CALIB_DIR  = KITTI_ROOT / "calib"

FRAMES = ["000046_10"]

OUT_DIR = Path(".") / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Depth / point cloud limits
Z_MIN, Z_MAX = 2.0, 60.0

# YOLO config
CONF_THRES = 0.35
ROI_PAD = 6  # small padding, not aggressive

# Conservative min points (400 is too strict for many KITTI ROIs)
MIN_POINTS_PER_OBJ = 150

# CALIB 
def read_kitti_calib_for_frame(frame_id: str):
    seq = frame_id.split("_")[0]
    calib_file = CALIB_DIR / f"{seq}.txt"
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    data = {}
    for line in calib_file.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = np.fromstring(v.strip(), sep=" ")

    P0_key = "P_rect_00" if "P_rect_00" in data else ("P0" if "P0" in data else None)
    P1_key = "P_rect_01" if "P_rect_01" in data else ("P1" if "P1" in data else None)
    if P0_key is None or P1_key is None:
        raise KeyError(f"Missing P_rect_00/P_rect_01 or P0/P1 in {calib_file}. Keys={list(data.keys())}")

    P0 = data[P0_key].reshape(3, 4)
    P1 = data[P1_key].reshape(3, 4)

    fx = float(P0[0, 0])
    fy = float(P0[1, 1])
    cx = float(P0[0, 2])
    cy = float(P0[1, 2])
    B  = float((P0[0, 3] - P1[0, 3]) / fx)

    return fx, fy, cx, cy, B


# STEREO 
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=192,  # must be divisible by 16
    blockSize=5,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
)

# YOLO 
def load_detector():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

def run_yolo(detector, bgr_img):
    results = detector.predict(source=bgr_img, verbose=False, conf=CONF_THRES)
    r = results[0]
    boxes = []
    if r.boxes is None:
        return boxes
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        conf = float(b.conf[0].cpu().numpy())
        cls  = int(b.cls[0].cpu().numpy())
        boxes.append((x1, y1, x2, y2, conf, cls))
    return boxes


# GEOMETRY 
def backproject_points(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.flatten()
    u = u_grid.flatten()
    v = v_grid.flatten()

    valid = np.isfinite(z) & (z > Z_MIN) & (z < Z_MAX)
    z = z[valid]
    u = u[valid]
    v = v[valid]

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    pts = np.vstack((X, Y, Z)).T
    return pts, u, v, h, w

def fit_aabb(pts):
    if pts is None or pts.shape[0] < 80:
        return None

    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 80:
        return None

    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

    x1, x2 = np.percentile(X, [5, 95])
    y1, y2 = np.percentile(Y, [5, 95])
    z1, z2 = np.percentile(Z, [5, 95])

    length = float(x2 - x1)
    height = float(y2 - y1)
    width  = float(z2 - z1)

    if length <= 0 or width <= 0 or height <= 0:
        return None
    if length > 25 or width > 12 or height > 8:
        return None

    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, (z1 + z2) / 2.0], dtype=np.float64)
    corners = np.array([
        [x1, y1, z1], [x2, y1, z1], [x2, y1, z2], [x1, y1, z2],
        [x1, y2, z1], [x2, y2, z1], [x2, y2, z2], [x1, y2, z2],
    ], dtype=np.float64)

    yaw = 0.0
    dims = (length, width, height)
    return center, dims, yaw, corners

def project_points_to_image(pts3d, fx, fy, cx, cy):
    X, Y, Z = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]
    Z = np.maximum(Z, 1e-6)
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return np.vstack([u, v]).T

def median_depth_gate(roi_pts, base_band=3.0):
    """
    Conservative depth gating:
    - Uses median +/- band
    - If it removes too many points, relax automatically
    """
    Z = roi_pts[:, 2]
    z_med = float(np.median(Z))

    # First pass: +/- base_band meters
    keep1 = (Z > z_med - base_band) & (Z < z_med + base_band)
    pts1 = roi_pts[keep1]

    # If too strict, relax
    if pts1.shape[0] < max(60, int(0.25 * roi_pts.shape[0])):
        keep2 = (Z > z_med - 6.0) & (Z < z_med + 6.0)
        pts2 = roi_pts[keep2]
        return pts2

    return pts1


# MAIN 
def process_frame(frame_id: str, detector):
    fx, fy, cx, cy, B = read_kitti_calib_for_frame(frame_id)
    print(f"\n=== Frame {frame_id} ===")
    print(f"[calib] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} B={B:.4f}m")

    left_path  = IMAGE0_DIR / f"{frame_id}.png"
    right_path = IMAGE1_DIR / f"{frame_id}.png"

    left_gray  = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right_gray = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    left_bgr   = cv2.imread(str(left_path))

    if left_gray is None or right_gray is None or left_bgr is None:
        print("Could not load images.")
        return

    disp = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp <= 0.1] = np.nan

    depth = fx * B / disp
    depth = np.clip(depth, 0, 80)
    np.save(OUT_DIR / f"depth_{frame_id}.npy", depth)

    pts, u, v, h, w = backproject_points(depth, fx, fy, cx, cy)
    print(f"[pc] points={pts.shape[0]} in Z[{Z_MIN},{Z_MAX}]")

    detections = run_yolo(detector, left_bgr)
    keep_cls = {0, 1, 2, 3, 5, 7}
    detections = [d for d in detections if d[5] in keep_cls]
    print(f"[yolo] detections kept={len(detections)}")

    overlay = left_bgr.copy()
    boxes3d = []

    for di, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
        x1i = int(max(0, x1 - ROI_PAD))
        y1i = int(max(0, y1 - ROI_PAD))
        x2i = int(min(w - 1, x2 + ROI_PAD))
        y2i = int(min(h - 1, y2 + ROI_PAD))

        roi_mask = (u >= x1i) & (u <= x2i) & (v >= y1i) & (v <= y2i)
        roi_pts = pts[roi_mask]
        n0 = roi_pts.shape[0]

        # Debug header for each detection
        print(f"[det {di}] cls={cls} conf={conf:.2f} 2d=({x1i},{y1i})-({x2i},{y2i}) pts_raw={n0}")

        if roi_pts.shape[0] < MIN_POINTS_PER_OBJ:
            print(f"[det {di}] skip: too few raw points (<{MIN_POINTS_PER_OBJ})")
            continue

        # sanitize
        roi_pts = roi_pts[np.isfinite(roi_pts).all(axis=1)]
        n1 = roi_pts.shape[0]

        # sane bounds
        Xr, Yr, Zr = roi_pts[:, 0], roi_pts[:, 1], roi_pts[:, 2]
        roi_pts = roi_pts[(np.abs(Xr) < 80) & (np.abs(Yr) < 25) & (Zr > 1.5) & (Zr < 60)]
        n2 = roi_pts.shape[0]

        if n2 < MIN_POINTS_PER_OBJ:
            print(f"[det {di}] skip: too few after sane bounds ({n2})")
            continue

        # conservative depth gate with fallback relaxation
        roi_pts = median_depth_gate(roi_pts, base_band=3.0)
        n3 = roi_pts.shape[0]

        if n3 < 80:
            print(f"[det {di}] skip: too few after depth gate ({n3})")
            continue

        fit = fit_aabb(roi_pts)
        if fit is None:
            print(f"[det {di}] skip: AABB fit rejected (dims sanity)")
            continue

        center, dims, yaw, corners = fit
        boxes3d.append((center, dims, yaw, corners, (x1i, y1i, x2i, y2i, conf, cls)))
        print(f"[det {di}] OK: pts {n0}->{n1}->{n2}->{n3} dims(L,W,H)=({dims[0]:.2f},{dims[1]:.2f},{dims[2]:.2f})")

        # Draw 2D box
        cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.putText(
            overlay, f"cls={cls} {conf:.2f}",
            (x1i, max(0, y1i - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

        # Project and draw 3D cuboid
        uv = project_points_to_image(corners, fx, fy, cx, cy).astype(int)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for a, b in edges:
            xa, ya = uv[a]
            xb, yb = uv[b]
            cv2.line(overlay, (xa, ya), (xb, yb), (0, 0, 255), 2)

    print(f"[3d] boxes fitted={len(boxes3d)}")

    out_img = OUT_DIR / f"yolo_3dbox_overlay_{frame_id}.png"
    cv2.imwrite(str(out_img), overlay)
    print("Saved:", out_img)

    out_txt = OUT_DIR / f"boxes3d_{frame_id}.txt"
    with open(out_txt, "w") as f:
        for i, (center, dims, yaw, corners, det) in enumerate(boxes3d):
            x1, y1, x2, y2, conf, cls = det
            f.write(
                f"{i} cls={cls} conf={conf:.3f} "
                f"center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}) "
                f"dims(L,W,H)=({dims[0]:.3f},{dims[1]:.3f},{dims[2]:.3f}) "
                f"yaw={yaw:.3f} 2d=({x1},{y1},{x2},{y2})\n"
            )
    print("Saved:", out_txt)


if __name__ == "__main__":
    detector = load_detector()
    for fid in FRAMES:
        process_frame(fid, detector)
