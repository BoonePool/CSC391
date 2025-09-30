import cv2
import numpy as np

def apply_convolution(image, kernel):
    """
    Apply convolution to a single-channel image with a given kernel.
    No use of cv2.filter2D.
    """
    # Ensure kernel is numpy array
    kernel = np.array(kernel)
    k_size = kernel.shape[0]
    pad = k_size // 2

    # Pad the image (zero padding chosen here)
    padded = np.pad(image, pad, mode='constant', constant_values=0)

    # Output image
    output = np.zeros_like(image, dtype=np.float32)

    # Convolution loop
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+k_size, j:j+k_size]
            output[i, j] = np.sum(region * kernel)

    # Clip to valid range and return uint8
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


# Define kernels
box_filter = np.ones((3, 3), dtype=np.float32) / 9.0   # 3x3 averaging filter
sharpen_filter = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
edge_filter = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]], dtype=np.float32)
soblel_filter_vert = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)
soblel_filter_horz = np.array([[-1, -2, -1],
                             [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)
gaussian_filter = (1/16) * np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=np.float32)
emboss_filter = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]], dtype=np.float32)
# Capture from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
mode = 1
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert to grayscale

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply filters
    # box_result = apply_convolution(gray, box_filter)
    # sharp_result = apply_convolution(gray, sharpen_filter)
    # edge_result = apply_convolution(gray, edge_filter)

    # Stack results for comparison
    #

    # Show the result


    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if (cv2.waitKey(1) & 0xFF == ord('1')) or mode == 1:
        result = cv2.filter2D(gray, -1,  sharpen_filter)
        mode = 1
    if cv2.waitKey(1) & 0xFF == ord('2') or mode == 2:
        result = cv2.filter2D(gray, -1,  edge_filter)
        mode = 2
    if cv2.waitKey(1) & 0xFF == ord('3') or mode == 3:
        result = cv2.filter2D(gray, -1,  box_filter)
        mode = 3
    if cv2.waitKey(1) & 0xFF == ord('4') or mode == 4:
        result = cv2.filter2D(gray, -1,  soblel_filter_vert)
        mode = 4
    if cv2.waitKey(1) & 0xFF == ord('5') or mode == 5:
        result = cv2.filter2D(gray, -1,  soblel_filter_horz)
        mode = 5
    if cv2.waitKey(1) & 0xFF == ord('6') or mode == 6:
        result = cv2.filter2D(gray, -1,  gaussian_filter)
        mode = 6
    if cv2.waitKey(1) & 0xFF == ord('7') or mode == 7:
        result = cv2.filter2D(gray, -1,  emboss_filter)
        mode = 7

    combined = np.hstack([gray, result])
    cv2.imshow("regular and choice ", combined)

cap.release()
cv2.destroyAllWindows()
