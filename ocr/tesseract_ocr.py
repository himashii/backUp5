from PIL import Image
import pytesseract
from unidecode import unidecode
import cv2
import numpy as np


def get_ocr(path):
    img = cv2.imread('upload/' + path)

    # img = cv2.imread('test.png')
    # gray = get_grayscale(img)
    # thresh = thresholding(gray)
    # opening = opening(gray)
    # canny = canny(gray)

    # cv2.imshow('gray', gray)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('opening', opening)
    # cv2.imshow('canny', canny)

    h, w, c = img.shape
    custom_config = r'-l sin --psm 6'
    strings = pytesseract.image_to_string(img, config=custom_config)
    boxes = pytesseract.image_to_boxes(img, config=custom_config)
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    save_name = 'outputs/' + path
    # cv2.imwrite(save_name, img)

    return ''.join(e for e in strings if e.isalnum())


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
