import cv2
import numpy as np

# Detect recognized objects using specified classifier and threshold values
def detect(img):
    cascade = cv2.CascadeClassifier("haarcascade_pack.xml")
    rects = cascade.detectMultiScale(img, 1.3, 3, cv2.CASCADE_SCALE_IMAGE, (74,74))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

# Draw rectangle on img where objects are detected
def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 10)

# Load image
img = cv2.imread('stack2.jpg')

# Draw rectangle around recognized matches
rects, img = detect(img)
box(rects, img)

# Display
img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
