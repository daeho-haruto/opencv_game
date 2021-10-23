import cv2
import numpy as np
import time
import random
import pyautogui

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def input_timer(prompt, timeout_sec):
    import subprocess
    import sys
    import threading
    import locale

    class Local:
        # check if timeout occured
        _timeout_occured = False

        def on_timeout(self, process):
            self._timeout_occured = True
            process.kill()
            # clear stdin buffer (for linux)
            # when some keys hit and timeout occured before enter key press,
            # that input text passed to next input().
            # remove stdin buffer.
            try:
                import termios
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except ImportError:
                # windows, just exit
                pass

        def input_timer_main(self, prompt_in, timeout_sec_in):
            # print with no new line
            print(prompt_in, end="")

            # print prompt_in immediately
            sys.stdout.flush()

            # new python input process create.
            # and print it for pass stdout
            cmd = [sys.executable, '-c', 'print(input())']
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                timer_proc = threading.Timer(timeout_sec_in, self.on_timeout, [proc])
                try:
                    # timer set
                    timer_proc.start()
                    stdout, stderr = proc.communicate()

                    # get stdout and trim new line character
                    result = stdout.decode(locale.getpreferredencoding()).strip("\r\n")
                finally:
                    # timeout clear
                    timer_proc.cancel()

            # timeout check
            if self._timeout_occured is True:
                # move the cursor to next line
                print("")
                raise TimeoutError
            return result

    t = Local()
    return t.input_timer_main(prompt, timeout_sec)

class Object:
    def __init__(self, size=50):
        self.logo_org = cv2.imread('opencv_project\img\logo.png')
        self.size = size
        self.logo = cv2.resize(self.logo_org, (size, size))
        img2gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        _, logo_mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        self.logo_mask = logo_mask
        self.x = 295 # 0~590
        self.y = 10 # 0~430
        self.score = 0

    def insert_object(self, frame):
        roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
        roi[np.where(self.logo_mask)] = 0
        roi += self.logo
    
    def update_position(self, tresh):
        height, width = tresh.shape

        num_x = random.randint(0,590)
        num_y = random.randint(0,430)

        roi = tresh[self.y:self.y + self.size, self.x:self.x + self.size]
        check = np.any(roi[np.where(self.logo_mask)])
        
        if check :
            self.score += 1 
            self.y = num_y
            self.x = num_x
            pyautogui.click(650, 1000)
            pyautogui.typewrite('true', interval=0.1)
            pyautogui.press('enter') 
            return check
        else :
            try:
                put = input_timer('timer >> ',3)
                print(put)
            except TimeoutError as e:
                self.score -= 1
                self.y = num_y
                self.x = num_x
                print("time out")
            
def make_mask_bgr(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    low = np.array([100,80,2])
    high = np.array([126,255,255])

    img_mask = cv2.inRange(img_hsv, low, high)

    return img_mask

obj = Object()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    delta_frame = make_mask_bgr(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) 
    delta_frame = cv2.morphologyEx(delta_frame, cv2.MORPH_CLOSE, kernel, 1)

    thresh = cv2.threshold(delta_frame, 127, 255, cv2.THRESH_BINARY)[1] 
    thresh = cv2.dilate(thresh, None, iterations=2)
    # cv2.imshow("track", delta_frame)

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
