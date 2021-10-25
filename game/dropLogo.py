import cv2
import numpy as np


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, bg_frame = cap.read()
    bg_frame = cv2.flip(bg_frame, 1)

    cv2.imshow("Webcam", bg_frame)

    if cv2.waitKey(1) == ord('q'):
        break 

bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)

class Object:
    def __init__(self, size=50):
        self.logo_org = cv2.imread('game\img\logo.png')
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
            self.score += 1

        roi = tresh[self.y:self.y + self.size, self.x:self.x + self.size]
        check = np.any(roi[np.where(self.logo_mask)])
        if check:
            self.score -= 1 
            self.y = 0
            self.x = np.random.randint(0, width - self.size - 1)
        return check
        
obj = Object()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    delta_frame = cv2.absdiff(bg_gray, gray)
    thresh = cv2.threshold(delta_frame, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    # cv2.imshow("track", thresh)

    hit = obj.update_position(thresh)
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