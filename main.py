import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Q1
def histogram():
    img = cv2.imread("images/Q1.jpg", 1)
    img_con = cv2.imread("images/Q1_1.jpg", 1)

    img_q2 = cv2.imread("images/Q1_2.jpg", 1)
    img_q3 = cv2.imread("images/Q1_3.jpg", 1)

    histogram_q1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    histogram_const = cv2.calcHist([img_con], [0], None, [256], [0, 256])

    plt.plot(histogram_q1, label='Normal image')
    plt.plot(histogram_const, label="High Contrast Image")
    plt.title("Histogram of Image 1")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    histogram_q2 = cv2.calcHist([img_q2], [0], None, [256], [0, 256])
    histogram_q3 = cv2.calcHist([img_q3], [0], None, [256], [0, 256])

    plt.plot(histogram_q1, label='Image 1')
    plt.plot(histogram_q2, label='Image 2')
    plt.plot(histogram_q3, label='Image 3')
    plt.title("Different image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# (Q2)
def contrast():
    img = cv2.imread("images/Q2.jpg", 0)
    mean = np.mean(img)

    gamma_1 = math.log(0.5 * 255) / math.log(mean)
    gamma_2 = math.log(0.3 * 255) / math.log(mean)
    gamma_3 = math.log(0.75 * 255) / math.log(mean)

    img_gamma1 = np.power(img, gamma_1).clip(0, 255).astype(np.uint8)
    img_gamma2 = np.power(img, gamma_2).clip(0, 255).astype(np.uint8)
    img_gamma3 = np.power(img, gamma_3).clip(0, 255).astype(np.uint8)

    plt.axis('off')
    plt.title('Gamma effect (0.5)')
    plt.imshow(cv2.cvtColor(img_gamma1, cv2.COLOR_BGR2RGB))

    plt.show()

    plt.axis('off')
    plt.title('Gamma effect (0.3)')
    plt.imshow(cv2.cvtColor(img_gamma2, cv2.COLOR_BGR2RGB))

    plt.show()
    plt.axis('off')
    plt.title('Gamma effect (0.75)')
    plt.imshow(cv2.cvtColor(img_gamma3, cv2.COLOR_BGR2RGB))


# (Q3)
def calculate_histogram():
    img = cv2.imread("images/Q3.jpg", 1)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title("Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


# (Q5)
def equalize_and_darkening():
    xray_image = cv2.imread('images/Q5.jfif', 0)

    equalized_image = cv2.equalizeHist(xray_image)
    darkened_image = (xray_image * 0.5).astype('uint8')
    darkened_equalized_image = cv2.equalizeHist(darkened_image)

    plt.axis('off')
    plt.title('Equalized X-ray')
    plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))

    plt.show()
    plt.axis('off')
    plt.title('Darkened X-ray')
    plt.imshow(cv2.cvtColor(darkened_image, cv2.COLOR_BGR2RGB))

    plt.show()
    plt.axis('off')
    plt.title('Equalized Darkened X-ray')
    plt.imshow(cv2.cvtColor(darkened_equalized_image, cv2.COLOR_BGR2RGB))


# (Q9)
def contrast_stretching():
    xray_image = cv2.imread('images/Q5.jfif', 1)
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    xray_stretched = cv2.LUT(xray_image, table)

    plt.axis('off')
    plt.title('Original X-ray')
    plt.imshow(cv2.cvtColor(xray_image, cv2.COLOR_BGR2RGB))

    plt.show()

    plt.axis('off')
    plt.title('Stretched X-ray')
    plt.imshow(cv2.cvtColor(xray_stretched, cv2.COLOR_BGR2RGB))


# (Q10)
def sharpen_image():
    img = cv2.imread('images/Q10_1.jpg', 1)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp_image = cv2.filter2D(img, -1, kernel)

    img_2 = cv2.imread('images/Q10_2.jpg', 1)
    blur_image = cv2.medianBlur(img_2, 5)

    plt.axis('off')
    plt.title('Sharpened Image')
    plt.imshow(cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB))

    plt.show()

    plt.axis('off')
    plt.title('Blurred Image')
    plt.imshow(cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB))


# Main function to call different methods
equalize_and_darkening()
