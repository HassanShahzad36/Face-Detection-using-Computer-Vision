import cv2 as cv
import numpy as np

img = cv.imread("Image.jfif")
cv.imshow("Original Image",img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
haar_cascade = cv.CascadeClassifier("haar_face.xml")
facesInImage = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print(facesInImage)
print(len(facesInImage))

for i,j,k,l in facesInImage:
    cv.rectangle(img, (i,j),(i+k,j+l),(0,255,0),thickness = 2)
cv.imshow("Faces Detected" , img)
cv.waitKey(0)
