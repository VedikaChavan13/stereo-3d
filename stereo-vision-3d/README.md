# **3D Object Detection and Localization using Stereo Cameras**

## **Project Overview**

This project implements a **stereo-only 3D perception pipeline** for robotic and autonomous driving scenarios using **rectified stereo camera inputs**. The system estimates **object-level 3D bounding boxes** without relying on LiDAR or active depth sensors, using only passive stereo vision and camera calibration.

The pipeline focuses on **geometry-driven 3D reconstruction and analysis**, emphasizing interpretability and modular design. It is intended as a research-oriented perception system that demonstrates how dense depth, point clouds, and object-level 3D localization can be recovered from stereo images.

The project is designed as an **end-to-end, reproducible computer vision pipeline**, with extensive qualitative visualization and failure-case analysis rather than benchmark-driven optimization.

---

## **Dataset**

This project uses the **KITTI Scene Flow (Stereo Flow) dataset**.

* Rectified left and right stereo image pairs
* Per-sequence camera calibration files
* No ground-truth 3D bounding boxes provided for the testing split

**Important notes:**

* The dataset is **not included** in this repository due to size and licensing restrictions.
* Users must download the dataset separately from the official KITTI website:
  [https://www.cvlibs.net/datasets/kitti/](https://www.cvlibs.net/datasets/kitti/)

The testing split is used primarily for **qualitative evaluation**, visualization, and geometric plausibility analysis.

---

## **Pipeline Tasks**

The implemented system performs the following tasks:

### **Stereo Disparity and Depth Estimation**

Dense disparity maps are computed using a Semi-Global Matching–style approach (OpenCV StereoSGBM). Disparity is converted into metric depth using camera focal length and stereo baseline from calibration files.

### **3D Point Cloud Reconstruction**

Valid depth pixels are backprojected into the camera coordinate frame to form a dense 3D point cloud. Range filtering is applied to remove implausible or noisy depth values.

### **2D Object Detection**

A pre-trained YOLO object detector is applied to the left stereo image to obtain 2D object bounding boxes that serve as spatial priors for 3D clustering.

### **Object-Centric 3D Clustering**

3D points that project inside each 2D detection are extracted to form object-specific point clusters. Additional filtering is applied to reduce background contamination and stereo outliers.

### **Oriented 3D Bounding Box Estimation**

For each valid cluster, an oriented 3D bounding box is estimated using PCA-based yaw estimation in the ground (XZ) plane, followed by robust extent estimation.

### **Visualization**

The system produces:

* Depth overlays
* Dense point cloud visualizations
* Object-centric cluster views
* Projected 3D bounding boxes overlaid on the original image

These outputs correspond directly to the figures presented in the final report.

---

## **Hardware Requirements**

* CPU capable of running OpenCV and Python
* GPU is **optional** (used only for faster YOLO inference)

---

## **Software Requirements**

* Python 3.8+
* OpenCV
* NumPy
* Matplotlib
* Ultralytics YOLO

Exact package versions are listed in `requirements.txt`.

---

## **Reproducibility Notes**

* The pipeline operates on individual frames to facilitate debugging and analysis.
* Intermediate outputs (depth maps, point clouds, overlays) are saved to disk to avoid recomputation.
* Results may vary depending on stereo parameters, scene content, and hardware configuration.

---

## **Originality and Contributions**

This project is **not a reimplementation of an existing end-to-end stereo 3D detection system**. The contribution lies in:

* A modular, interpretable stereo-only 3D perception pipeline
* Object-centric 3D clustering using 2D detections as spatial priors
* PCA-based oriented 3D bounding box fitting from noisy stereo point clouds
* Detailed qualitative evaluation and failure-case analysis

The focus is on **analysis, system integration, and understanding practical limitations** of stereo-only 3D perception rather than achieving state-of-the-art benchmark performance.

---

## **Limitations**

* Stereo depth quality degrades for distant and low-texture objects
* Object clustering depends on the quality of 2D detections
* No temporal consistency or tracking is applied
* Evaluation is qualitative due to lack of ground-truth 3D boxes in the dataset split

The system is intended as a **research prototype**, not a production-ready perception module.

---

## **Author**

**Vedika Chavan**
Master of Science in Computer Science
Department of Computer Science
Binghamton University, State University of New York


