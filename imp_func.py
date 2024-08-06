import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
def invert(image):
    return cv2.bitwise_not(image)
'''

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresh(image):
    return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

def remove_noise(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def segment_lines(binary_image, original_image, min_width=30, max_width=1000, min_height=40, max_height=200):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    
    lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if min_width <= w <= max_width and min_height <= h <= max_height:
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            line_image = original_image[y:y + h, x:x + w]
            lines.append(line_image)
    
    return original_image, lines


def preprocessed_image(image):
    gray_image = grayscale(image)
    binary_image = thresh(gray_image)










'''
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray_image, binary_image

def seperate(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
    characters = [binary_image[y:y + h, x:x + w] for (x, y, w, h) in bounding_boxes]
    return characters

def preprocess_char_image(char_image):
    char_image = cv2.resize(char_image, (28, 28))
    char_image = char_image.astype('float32') / 255.0
    char_image = np.expand_dims(char_image, axis=-1)
    char_image = np.expand_dims(char_image, axis=0)
    return char_image
'''

def display(image):
    dpi = 80
    height, width  = image.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image, cmap='gray')
    plt.show()