import cv2
import numpy as np
import matplotlib.pyplot as plt

# load images
original_image = cv2.imread("C:\\Users\\Boone Pool\\Downloads\\original.jpg")  
transformed_image = cv2.imread("C:\\Users\\Boone Pool\\Downloads\\transformed_image.jpg") 


# Convert for display purposes (BGR -> RGB)
orig_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
transformed = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)


rows, cols = original_image.shape[:2]

#Perspective (used the matlab veiwer to get the points)
pts1 = np.float32([[0,0],[1200,0],[284,916]])  # source square
pts2 = np.float32([[300,55],[891,490],[800,1082]])  # mapped quad
M_perspective = cv2.getAffineTransform(pts1, pts2)
perspective = cv2.warpAffine(original_image, M_perspective, (cols, rows))


perspective_rgb = cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB)



print("Applied transformations:")
print("Affine transform matrix:\n", M_perspective)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(orig_rgb)
axs[0].set_title("Original")
axs[0].axis("off")

axs[1].imshow(perspective_rgb)
axs[1].set_title("Transformed (Perspective)")
axs[1].axis("off")

axs[2].imshow(transformed)
axs[2].set_title("transformed_image")
axs[2].axis("off")

plt.tight_layout()
plt.show()