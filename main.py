import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


#Q1
def histogram():
    img = cv2.imread("images/Q1.jpg", 1)
    img_con = cv2.imread("images/Q1_1.jpg", 1)

    img_q2 = cv2.imread("images/Q1_2.jpg", 1)
    img_q3 = cv2.imread("images/Q1_3.jpg", 1)

    histogram_q1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    histogram_const = cv2.calcHist([img_con], [0], None, [256], [0, 256])

    # Plot the histogram
    plt.plot(histogram_q1, label='Normal image')
    plt.plot(histogram_const, label="High Contrast Image")
    plt.title("Histogram Image 1")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    histogram_q2 = cv2.calcHist([img_q2], [0], None, [256], [0, 256])
    histogram_q3 = cv2.calcHist([img_q3], [0], None, [256], [0, 256])

    plt.plot(histogram_q1, label='Picture 1')
    plt.plot(histogram_q2, label='Picture 2')
    plt.plot(histogram_q3, label='Picture 3')
    plt.title("Diffrent image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def contrast():
    img = cv2.imread("images/Q2.jpg", 0)
    mean = np.mean(img)

    gamma_1 = math.log(0.5 * 255) / math.log(mean)
    gamma_2 = math.log(0.3 * 255) / math.log(mean)
    gamma_3 = math.log(0.75 * 255) / math.log(mean)

    img_gamma1 = np.power(img, gamma_1).clip(0, 255).astype(np.uint8)
    img_gamma2 = np.power(img, gamma_2).clip(0, 255).astype(np.uint8)
    img_gamma3 = np.power(img, gamma_3).clip(0, 255).astype(np.uint8)

    cv2.imshow("Gamma effect (0.5)", img_gamma1)
    cv2.imshow("Gamma effect (0.3)", img_gamma2)
    cv2.imshow("Gamma effect (0.75)", img_gamma3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Method for Histogram Calculation (Q3)
def calculate_histogram(image_path):
    img = cv2.imread(image_path, 1)  # Read the image in grayscale
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title("Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


# Method for Histogram Equalization and Darkening (Q5)
def equalize_and_darkening():
    xray_image = cv2.imread('images/Q5.jfif', 0)

    # Equalize the image
    equalized_image = cv2.equalizeHist(xray_image)
    # Darken the image
    darkened_image = (xray_image * 0.5).astype('uint8')
    # Equalize the darkened image
    darkened_equalized_image = cv2.equalizeHist(darkened_image)

    # Display the results
    cv2.imshow("Equalized X-ray", equalized_image)
    cv2.imshow("Darkened X-ray", darkened_image)
    cv2.imshow("Equalized Darkened X-ray", darkened_equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Method for Contrast Stretching (Q9)
def contrast_stretching(image_path):
    xray_image = cv2.imread(image_path, 1)
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    xray_stretched = cv2.LUT(xray_image, table)

    # Display the results
    cv2.imshow("Original X-ray", xray_image)
    cv2.imshow("Stretched X-ray", xray_stretched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Method for Sharpening (Q10)
def sharpen_image(image_path):
    img = cv2.imread(image_path, 1)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    im = cv2.filter2D(img, -1, kernel)

    # Display the sharpened image
    cv2.imshow("Sharpened Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blurry_image(image_path):
    img = cv2.imread(image_path, 1)
    im = cv2.medianBlur(img, 5)

    # Display the sharpened image
    cv2.imshow("Blurred Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to call different methods
equalize_and_darkening()

