import cv2
import timeit
import numpy as np

def detect(cam, cascade):
    rects = cascade.detectMultiScale(cam, scaleFactor= 1.1,
                                           minNeighbors=5, 
                                           minSize=(30,30),
                                           flags = cv2.CASCADE_SCALE_IMAGE
                                           )
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def removeFace(cam, cascade):
    gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)    
    gray = cv2.equalizeHist(gray)
    rects = detect(gray,cascade)

    height = width = cam.shape[0]

    for x1 ,y1, x2 , y2 in rects:
        cv2.rectangle(cam,(x1-15,0), (x2+15, height), (0,0,0), -1)
    return cam

def make_mask_bgr(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    low = np.array([100,80,2])
    high = np.array([126,255,255])

    img_mask = cv2.inRange(img_hsv, low, high)

    # mask = cv2.bitwise_and(img_bgr, img_bgr,mask=img_mask)
    return img_mask
    
def findMaxArea(contours):
    max_contour = None
    max_area = -1

    for contour in contours:
        area = cv2.contourArea(contour)

        x,y,w,h = cv2.boundingRect(contour)

        if (w*h)*0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = contour
  
    if max_area < 10000:
        max_area = -1

    return max_area, max_contour

cascade_filename = 'opencv_project\haarcascade_frontalface_alt.xml'

cascade = cv2.CascadeClassifier(cascade_filename)

cap = cv2.VideoCapture(0)

while True:
    _,first = cap.read()
    _,face = cap.read()

    color = removeFace(face,cascade)
    color = make_mask_bgr(color)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) 
    color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, kernel, 1)

    ret1, thr = cv2.threshold(color, 127,255,0)

    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    area, contour = findMaxArea(contours)
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(first,(x,y), (x+w, h+y), (255,255,255), 5)

    
    cv2.imshow('First',first)
    cv2.imshow('RemoveFace',face)
    cv2.imshow('ColorExtraction',color)

    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()