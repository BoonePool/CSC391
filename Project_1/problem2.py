import rawpy
import numpy as np
import math
from matplotlib import pyplot as plt

# Function to calculate horizontal Field of View (FOV)
def calculate_fov(sensor_width, focal_length):
    return 2 * math.atan(sensor_width / (2 * focal_length))

# Function to calculate noise (mean and standard deviation) in a selected region
def calculate_noise(image, x, y, width, height):
    region = image[y:y+height, x:x+width]
    mean = np.mean(region)
    std_dev = np.std(region)
    return mean, std_dev

# Load raw images
def load_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        return raw.postprocess()

# Camera specifications (replace with your phone's values)
main_camera_sensor_width = 6  # in mm
main_camera_focal_length = 26  # in mm
wide_camera_sensor_width =  5.6 # in mm
wide_camera_focal_length = 13  # in mm


# Calculate FOV for both cameras
main_camera_fov = calculate_fov(main_camera_sensor_width, main_camera_focal_length)
telephoto_camera_fov = calculate_fov(wide_camera_sensor_width, wide_camera_focal_length)

print(f"Main Camera FOV: {math.degrees(main_camera_fov):.2f} degrees")
print(f"wide Camera FOV: {math.degrees(telephoto_camera_fov):.2f} degrees")

# Load images
main_camera_image = load_raw_image("C:\\Users\\Boone Pool\\Downloads\\APC_0004.dng")
telephoto_camera_image = load_raw_image("C:\\Users\\Boone Pool\\Downloads\\APC_0003.dng")

# Display images to select a region for noise analysis
plt.imshow(main_camera_image)
plt.title("Main Camera Image")
plt.show()

plt.imshow(telephoto_camera_image)
plt.title("wide Camera Image")
plt.show()

# Define region for noise analysis (manually set coordinates and size)
# Replace x, y, width, height with actual values
x, y, width, height = 115, 2000, 50, 50

# Calculate noise for both images
main_mean, main_std = calculate_noise(main_camera_image, x, y, width, height)
x, y, width, height = 50, 2000, 50, 50
telephoto_mean, telephoto_std = calculate_noise(telephoto_camera_image, x, y, width, height)

print(f"Main Camera Noise - Mean: {main_mean:.2f}, Std Dev: {main_std:.2f}")
print(f"wide Camera Noise - Mean: {telephoto_mean:.2f}, Std Dev: {telephoto_std:.2f}")