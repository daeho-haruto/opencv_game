import cv2
import numpy as np


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, bg_frame = cap.read()
    bg_frame = cv2.flip(bg_frame, 1)

    cv2.imshow("Webcam", bg_frame)

    if cv2.waitKey(1) == ord('q'):
        break 

class Object:
    def __init__(self, size=50):
        self.logo_org = cv2.imread('opencv_project\img\logo.png')
        self.size = size
        self.logo = cv2.resize(self.logo_org, (size, size))
        img2gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        _, logo_mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        self.logo_mask = logo_mask
        self.speed = 10
        self.x = 0
        self.y = 0
        self.score = 0

    def insert_object(self, frame):
        roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
        roi[np.where(self.logo_mask)] = 0
        roi += self.logo

    def update_position(self, tresh):
        height, width = tresh.shape
        self.y += self.speed
        if self.y + self.size > height:
            self.y = 0
            self.x = np.random.randint(0, width - self.size - 1)
            self.score -= 1

        roi = tresh[self.y:self.y + self.size, self.x:self.x + self.size]
        check = np.any(roi[np.where(self.logo_mask)])
        if check:
            self.score += 1 
            self.y = 0
            self.x = np.random.randint(0, width - self.size - 1)
        return check
        
obj = Object()

def make_mask_bgr(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    low = np.array([100,80,2])
    high = np.array([126,255,255])

    img_mask = cv2.inRange(img_hsv, low, high)

    return img_mask

while True:

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    delta_frame = make_mask_bgr(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) 
    delta_frame = cv2.morphologyEx(delta_frame, cv2.MORPH_CLOSE, kernel, 1)
    
    thresh = cv2.threshold(delta_frame, 127, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cv2.imshow("track", delta_frame)

    hit = obj.update_position(delta_frame)
    obj.insert_object(frame)

    if hit:
        frame[:, :, :] = 255

    text = f"Score: {obj.score}"
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()