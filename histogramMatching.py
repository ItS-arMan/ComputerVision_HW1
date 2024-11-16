import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_input = cv.imread('images/Q4.jpg')
img_ref = cv.imread('images/Q4_ref.jpg')


def calculate_cdf(hist):
    cdf = np.cumsum(hist)
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalize to range [0, 255]
    return cdf_normalized


# Calculate histograms and CDFs for input and reference images
hist_input = cv2.calcHist([img_input], [0], None, [256], [0, 256])
cdf_input = calculate_cdf(hist_input)

hist_ref = cv2.calcHist([img_ref], [0], None, [256], [0, 256])
cdf_ref = calculate_cdf(hist_ref)

# Create a mapping from the input image to the reference image
mapping = np.zeros(256)
for i in range(256):
    # Find the closest match in the reference CDF
    diff = np.abs(cdf_ref - cdf_input[i])
    mapping[i] = np.argmin(diff)

# Apply the mapping to create the matched image
matched_image = np.zeros_like(img_input)
for i in range(img_input.shape[0]):
    for j in range(img_input.shape[1]):
        matched_image[i, j] = mapping[img_input[i, j]]

plt.figure()
plt.title('Matched Image')
plt.imshow(matched_image)
plt.show()