import cv2
from matplotlib import pyplot as plt

# Load left and right images
left_img = cv2.imread("left.png")
right_img = cv2.imread("right.png")

# Check that images loaded
if left_img is None or right_img is None:
    print("One or both images not found. Check file names!")
else:
    # Convert from BGR (OpenCV default) to RGB for plotting
    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    # Display side by side
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Left Image")
    plt.imshow(left_rgb)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Right Image")
    plt.imshow(right_rgb)
    plt.axis("off")
    plt.show()

