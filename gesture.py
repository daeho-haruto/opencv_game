import cv2
import pyautogui
import numpy as np

cap = cv2.VideoCapture(0)
PINK_MIN = np.array([100,80,2], np.uint8)
PINK_MAX = np.array([126,255,255], np.uint8)

centroid_x = 0
centroid_y = 0
s = ''
move = ''

while(True):

    ret, img = cap.read()
    img = cv2.flip(img, 1)

    #thresh = cv2.namedWindow('Threshold', cv2.WINDOW_NORMAL)
    orig = cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    #img = cv2.GaussianBlur(img, (15, 15), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    #grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #_, frame_threshed = cv2.threshold(grey, 127, 255,
    #                        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    frame_threshed = cv2.inRange(hsv, PINK_MIN, PINK_MAX)

    contours,hierarchy = cv2.findContours(frame_threshed, 1, 2)
    max_area = 0
    last_x = centroid_x
    last_y = centroid_y

    if contours:
        for i in contours:
            area = cv2.contourArea(i)
            if area > max_area:
                max_area = area
                cnt = i

        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        centroid_x = int((x + x+w)/2)
        centroid_y = int((y + y+h)/2)

        cv2.circle(img, (centroid_x, centroid_y), 2, (0,0,255), 2)

        cv2.line(img,(196,0),(196,480),(0,255,0),5)
        cv2.line(img,(392,0),(392,480),(0,255,0),5)
        cv2.line(img,(196,240),(392, 240),(0,255,0),5)

        #cv2.imshow('Threshold', frame_threshed)
        cv2.imshow('Original', img)


 
        # up-down move
        if centroid_x >= 196 and centroid_x <= 392:
            # up
            if centroid_y >= 0 and centroid_y <= 240:
                print ('up')
                # pyautogui.press('up')
            # down
            if centroid_y >= 240 and centroid_y <=480:
                print ('down')
                # pyautogui.press('down')

        # left-right move
        if centroid_y >= 0 and centroid_y <= 480:
            # left
            if centroid_x >= 0 and centroid_x <= 196:
                print ('left')
                # pyautogui.press('left')
            # right
            if centroid_x >= 392:
                print ('right')
                # pyautogui.press('right')

        ##right-left move
        #diff_x = centroid_x - last_x
        #if diff_x >= 30:
        #    print 'right'
        #    pyautogui.press('right')
        #    s = 'right'
        #elif diff_x <= -30:
        #    print 'left'
        #    pyautogui.press('left')
        #    s = 'left'

        ##up-down move
        #diff_y = centroid_y - last_y
        #if diff_y >= 30:
        #    print 'down'
        #    pyautogui.press('down')
        #    s = 'down'
        #elif diff_y <= -30:
        #    print 'up'
        #    pyautogui.press('up')
        #    s = 'up'
        #move = s

    k = cv2.waitKey(10)
    if k == 27:
        break
