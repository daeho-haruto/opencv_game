import cv2
import pygame
import mediapipe as mp
import numpy as np
import time
import random

max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok'
}
rps_gesture = {0:'zero', 1:'one', 9:'two', 3:'three', 4:'four', 5:'five'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('game\data\gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

class GUI:
    size = (0, 0)
    screen = None
    score = 0
    def __init__(self):
        self.home()

    def home(self):
        pygame.init()  # 초기화
        self.size = (pygame.display.Info().current_w, pygame.display.Info().current_h)
        self.screen = pygame.display.set_mode(self.size, pygame.FULLSCREEN)
        pygame.display.set_caption("Game Title")  # 타이틀
        self.screen.fill((0, 0, 0))  # BG color
        self.Update()
    
    def Update(self):
        gameEnd = False
        last_state = None
        state_cnt = 0
        while not gameEnd:
            pygame.time.Clock().tick(30)
            rect, img = cap.read()
            if not rect:
                continue
            img = cv2.resize(img,self.size)
            img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(img), 270), (0, 0))
            result = hands.process(img)
            self.screen.blit(pygame.font.SysFont('malgungothic', 36).render("메인화면", True, (0, 0, 0)), (780, 0))
            if result.multi_hand_landmarks is not None:
                rps_result = []
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                    v = v2 - v1 # [20,3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    # Inference gesture
                    data = np.array([angle], dtype=np.float32)
                    ret, results, neighbours, dist = knn.findNearest(data, 3)
                    idx = int(results[0][0])

                    # Draw gesture result
                    if idx in rps_gesture.keys():
                        org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                        
                        rps_result.append({
                            'rps': rps_gesture[idx],
                            'org': org
                        })
                if rps_result:
                    state = rps_result[0]['rps']
                    self.screen.blit(pygame.font.SysFont('malgungothic', 68).render("GAME : "+{'zero':'0','one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5'}[state], True, (40, 255, 30)), (0, 0))
                    if state == last_state:
                        state_cnt += 1
                    else:
                        state_cnt = 0
                    if state_cnt >= 30:
                        if state == 'one':
                            self.dropLogo()
                            self.showScore()
                            state_cnt = 0
                        elif state == 'two':
                            self.catchLogo()
                            self.showScore()
                            state_cnt = 0
                        elif state == 'three':
                            self.pingpong()
                            self.showScore()
                            state_cnt = 0
                        elif state == 'four':
                            self.pingpong2()
                            self.showScore()
                            state_cnt = 0
                    
                    last_state = state
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    gameEnd = True
            pygame.display.flip()

    def showScore(self):
        is_showing = True
        s_time = time.time()
        while is_showing:
            _, bg_frame = cap.read()
            bg_frame = cv2.resize(bg_frame,self.size)
            bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
            e_time = time.time()
            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(bg_frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('None', 72).render('Score : '+str(self.score), True, (40, 255, 30)), (0, 0))
            if e_time - s_time >= 5:
                break
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_showing = False
            pygame.display.flip()

    def dropLogo(self):
        is_playing = True
        while is_playing:
            _, bg_frame = cap.read()
            bg_frame = cv2.resize(bg_frame,self.size)
            bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
            bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
            
            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(bg_frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('malgungothic', 72).render("캠 화면에서 나가주시고 'q' 를 눌러주세요", True, (255, 0, 0)), (0, 0))
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        is_playing = False

            pygame.display.flip()
        
        is_playing = True 
        s_time = time.time()
        while is_playing:
            _, ready_frame = cap.read()
            ready_frame = cv2.resize(ready_frame ,self.size)
            ready_frame = cv2.cvtColor(ready_frame, cv2.COLOR_BGR2RGB)

            e_time = time.time()

            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(ready_frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('malgungothic', 72).render(str(10-int(e_time-s_time)) + "초 뒤에 시작합니다.", True, (255, 0, 0)), (0, 0))

            if e_time - s_time >= 10 :
                is_playing = False
            
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        is_playing = False
            pygame.display.flip()

        class Object:
            def __init__(self, size=150):
                self.logo_org = cv2.imread('game\img\logo.png')
                self.size = size
                self.logo = cv2.resize(self.logo_org, (size, size))
                img2gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
                _, logo_mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                self.logo_mask = logo_mask
                self.speed = 40
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

        is_playing = True
        start_time = time.time()
        while is_playing:
            _, frame = cap.read()
            frame = cv2.resize(frame,self.size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            delta_frame = cv2.absdiff(bg_gray, gray)
            thresh = cv2.threshold(delta_frame, 100, 255, cv2.THRESH_BINARY)[1]
            thres= cv2.dilate(thresh, None, iterations=2)
            
            hit = obj.update_position(thresh)
            obj.insert_object(frame)

            if hit:
                frame[:, :, :] = 255

            text = f"Score: {obj.score}"

            end_time = time.time()

            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('None', 72).render(text,True, (0, 255, 0)), (0, 0))
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(30-int(end_time-start_time)),True, (0, 255, 0)), (self.size[0]-60, 0))

            if end_time - start_time >= 30:
                is_playing = False
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False

            pygame.display.flip()
        self.score = obj.score
        
    
    def catchLogo(self):
        is_playing = True 
        s_time = time.time()
        while is_playing:
            _, ready_frame = cap.read()
            ready_frame = cv2.resize(ready_frame ,self.size)
            ready_frame = cv2.cvtColor(ready_frame, cv2.COLOR_BGR2RGB)

            e_time = time.time()

            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(ready_frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('malgungothic', 72).render(str(10-int(e_time-s_time)) + "초 뒤에 시작합니다.", True, (255, 0, 0)), (0, 0))

            if e_time - s_time >= 10 :
                is_playing = False
            
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        is_playing = False
            pygame.display.flip()


        class Object:
            def __init__(self, size=100):
                self.logo_org = cv2.imread('game\img\logo.png')
                self.size = size
                self.logo = cv2.resize(self.logo_org, (size, size))
                img2gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
                _, logo_mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                self.logo_mask = logo_mask
                self.x = 800
                self.y = 100
                self.score = 0

            def insert_object(self, frame):
                roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
                roi[np.where(self.logo_mask)] = 0
                roi += self.logo
            
            def update_position(self, tresh):
                num_x = random.randint(0,1600)
                num_y = random.randint(0,950)

                roi = tresh[self.y:self.y + self.size, self.x:self.x + self.size]
                check = np.any(roi[np.where(self.logo_mask)])

                if check :
                    self.score += 1
                    self.y = num_y
                    self.x = num_x
                
                return check
                    
        def make_mask_bgr(img_bgr):

            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            low = np.array([100,140,100])
            high = np.array([130,255,255])

            img_mask = cv2.inRange(img_hsv, low, high)

            return img_mask

        obj = Object()

        is_playing = True
        start_time = time.time()
        while is_playing:
            _, frame = cap.read()
            frame = cv2.resize(frame,self.size)
            delta_frame = make_mask_bgr(frame)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) 
            delta_frame = cv2.morphologyEx(delta_frame, cv2.MORPH_CLOSE, kernel, 1)

            thresh = cv2.threshold(delta_frame, 127, 255, cv2.THRESH_BINARY)[1] 
            thresh = cv2.dilate(thresh, None, iterations=2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hit = obj.update_position(delta_frame)
            end_time = time.time()

            last_time = int(end_time-start_time)

            obj.insert_object(frame)

            if hit:
                frame[:, :, :] = 255
            
            text = f"Score: {obj.score}"

            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(text), True, (0, 255, 0)), (0, 0))
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(30-last_time), True, (0, 255, 0)), (self.size[0]-60, 0))

            if last_time == 30:
                is_playing = False
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False
            
            pygame.display.flip()

        self.score = obj.score
        
    
    def pingpong(self):
        is_playing = True 
        s_time = time.time()
        while is_playing:
            _, ready_frame = cap.read()
            ready_frame = cv2.resize(ready_frame ,self.size)
            ready_frame = cv2.cvtColor(ready_frame, cv2.COLOR_BGR2RGB)

            e_time = time.time()

            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(ready_frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('malgungothic', 72).render(str(10-int(e_time-s_time)) + "초 뒤에 시작합니다.", True, (255, 0, 0)), (0, 0))

            if e_time - s_time >= 10 :
                is_playing = False
            
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        is_playing = False
            pygame.display.flip()

        
        bar = pygame.Surface((10,50))
        bar1 = bar.convert()
        bar1.fill((0,0,255))
        bar2 = bar.convert()
        bar2.fill((255,0,0))
        circ_sur = pygame.Surface((15,15))
        circ = pygame.draw.circle(circ_sur,(0,255,0),(15/2,15/2),15/2)
        circle = circ_sur.convert()
        circle.set_colorkey((0,0,0))


        bar1_x, bar2_x = 10. , 620.
        bar1_y, bar2_y = 215. , 215.
        circle_x, circle_y = 307.5, 232.5
        bar1_move, bar2_move = 0. , 0.
        speed_x, speed_y, speed_circ = 250., 250., 250.
        bar1_score, bar2_score = 0,0

        clock = pygame.time.Clock()
        font = pygame.font.SysFont("calibri",40)

        ai_speed = 0


        is_playing = True
        s_time = time.time()
        while is_playing:
            rect, img = cap.read()
            if not rect:
                continue
            img = cv2.resize(img,self.size)
            img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)


            result = hands.process(img)
            if result.multi_hand_landmarks is not None:
                rps_result = []
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                    v = v2 - v1 # [20,3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    # Inference gesture
                    data = np.array([angle], dtype=np.float32)
                    ret, results, neighbours, dist = knn.findNearest(data, 3)
                    idx = int(results[0][0])

                    # Draw gesture result
                    if idx in rps_gesture.keys():
                        org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                        
                        rps_result.append({
                            'rps': rps_gesture[idx],
                            'org': org
                        })
                if rps_result:
                    state = rps_result[0]['rps']
                    if state == 'zero':
                        bar1_move = ai_speed
                    elif state == 'five':
                        bar1_move = -ai_speed

            score1 = font.render(str(bar1_score), True,(255,255,255))
            score2 = font.render(str(bar2_score), True,(255,255,255))
            
            self.screen.fill((0, 0, 0))
            # self.screen.blit(background,(0,0))
            frame = pygame.draw.rect(self.screen,(255,255,255), pygame.Rect((5,5),(630,470)),2)
            middle_line = pygame.draw.aaline(self.screen,(255,255,255),(330,5),(330,475))
            self.screen.blit(bar1,(bar1_x,bar1_y))
            self.screen.blit(bar2,(bar2_x,bar2_y))
            self.screen.blit(circle,(circle_x,circle_y))
            self.screen.blit(score1,(250.,210.))
            self.screen.blit(score2,(380.,210.))

            e_time = time.time()
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(60-int(e_time-s_time)), True, (0, 255, 0)), (640, 0))

            bar1_y += bar1_move
    

            time_passed = clock.tick(30)
            time_sec = time_passed / 1000.0
            
            circle_x += speed_x * time_sec
            circle_y += speed_y * time_sec 
            ai_speed = speed_circ * time_sec

            if circle_x >= 305.:
                if not bar2_y == circle_y + 7.5:
                    if bar2_y < circle_y + 7.5:
                        bar2_y += ai_speed
                    if  bar2_y > circle_y - 42.5:
                        bar2_y -= ai_speed
                else:
                    bar2_y == circle_y + 7.5
            
            if bar1_y >= 420.: bar1_y = 420.
            elif bar1_y <= 10. : bar1_y = 10.
            if bar2_y >= 420.: bar2_y = 420.
            elif bar2_y <= 10.: bar2_y = 10.

            if circle_x <= bar1_x + 10.:
                if circle_y >= bar1_y - 7.5 and circle_y <= bar1_y + 42.5:
                    circle_x = 20.
                    speed_x = -speed_x
            if circle_x >= bar2_x - 15.:
                if circle_y >= bar2_y - 7.5 and circle_y <= bar2_y + 42.5:
                    circle_x = 605.
                    speed_x = -speed_x
            if circle_x < 5.:
                bar2_score += 1
                circle_x, circle_y = 320., 232.5
                bar1_y,bar_2_y = 215., 215.
            elif circle_x > 620.:
                bar1_score += 1
                circle_x, circle_y = 307.5, 232.5
                bar1_y, bar2_y = 215., 215.
            if circle_y <= 10.:
                speed_y = -speed_y
                circle_y = 10.
            elif circle_y >= 457.5:
                speed_y = -speed_y
                circle_y = 457.5

            if e_time - s_time >= 60:
                is_playing = False
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False
            
            pygame.display.flip()
        self.score = str(bar1_score) + " : " + str(bar2_score)

    def pingpong2(self):
        is_playing = True 
        s_time = time.time()
        while is_playing:
            _, ready_frame = cap.read()
            ready_frame = cv2.resize(ready_frame ,self.size)
            ready_frame = cv2.cvtColor(ready_frame, cv2.COLOR_BGR2RGB)

            e_time = time.time()

            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(ready_frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('malgungothic', 72).render(str(10-int(e_time-s_time)) + "초 뒤에 시작합니다.", True, (255, 0, 0)), (0, 0))

            if e_time - s_time >= 10 :
                is_playing = False
            
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        is_playing = False
            pygame.display.flip()

            
        PINK_MIN = np.array([100,160,140], np.uint8)
        PINK_MAX = np.array([126,255,255], np.uint8)

        centroid_x = 0
        centroid_y = 0
        s = ''
        move = ''

        
        bar = pygame.Surface((10,50))
        bar1 = bar.convert()
        bar1.fill((0,0,255))
        bar2 = bar.convert()
        bar2.fill((255,0,0))
        circ_sur = pygame.Surface((15,15))
        circ = pygame.draw.circle(circ_sur,(0,255,0),(15/2,15/2),15/2)
        circle = circ_sur.convert()
        circle.set_colorkey((0,0,0))


        bar1_x, bar2_x = 10. , 620.
        bar1_y, bar2_y = 215. , 215.
        circle_x, circle_y = 307.5, 232.5
        bar1_move, bar2_move = 0. , 0.
        speed_x, speed_y, speed_circ = 250., 250., 250.
        bar1_score, bar2_score = 0,0

        clock = pygame.time.Clock()
        font = pygame.font.SysFont("calibri",40)

        ai_speed = 0


        is_playing = True
        s_time = time.time()
        cnt = 0
        while is_playing:
            rect, img = cap.read()
            img = cv2.flip(img, 1)
            if not rect:
                continue

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

                cv2.line(img,(0,240),(640, 240),(0,255,0),5)

                # cv2.imshow('Threshold', frame_threshed)
                # img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)

                cv2.imshow('control',img)

                # self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(img), 270), (0, 0))
                # up-down move
                # up
                if centroid_y >= 0 and centroid_y <= 240:
                    bar1_move = -ai_speed
                # down
                if centroid_y >= 240 and centroid_y <=480:
                    bar1_move = ai_speed
 

            score1 = font.render(str(bar1_score), True,(255,255,255))
            score2 = font.render(str(bar2_score), True,(255,255,255))
            
            self.screen.fill((0, 0, 0))
            # self.screen.blit(background,(0,0))
            frame = pygame.draw.rect(self.screen,(255,255,255), pygame.Rect((5,5),(630,470)),2)
            middle_line = pygame.draw.aaline(self.screen,(255,255,255),(330,5),(330,475))
            self.screen.blit(bar1,(bar1_x,bar1_y))
            self.screen.blit(bar2,(bar2_x,bar2_y))
            self.screen.blit(circle,(circle_x,circle_y))
            self.screen.blit(score1,(250.,210.))
            self.screen.blit(score2,(380.,210.))

            e_time = time.time()
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(60-int(e_time-s_time)), True, (0, 255, 0)), (640, 0))

            bar1_y += bar1_move
    

            time_passed = clock.tick(30)
            time_sec = time_passed / 1000.0
            
            circle_x += speed_x * time_sec
            circle_y += speed_y * time_sec 
            ai_speed = speed_circ * time_sec

            if circle_x >= 305.:
                if not bar2_y == circle_y + 7.5:
                    if bar2_y < circle_y + 7.5:
                        bar2_y += ai_speed
                    if  bar2_y > circle_y - 42.5:
                        bar2_y -= ai_speed
                else:
                    bar2_y == circle_y + 7.5
            
            if bar1_y >= 420.: bar1_y = 420.
            elif bar1_y <= 10. : bar1_y = 10.
            if bar2_y >= 420.: bar2_y = 420.
            elif bar2_y <= 10.: bar2_y = 10.

            if circle_x <= bar1_x + 10.:
                if circle_y >= bar1_y - 7.5 and circle_y <= bar1_y + 42.5:
                    circle_x = 20.
                    speed_x = -speed_x
            if circle_x >= bar2_x - 15.:
                if circle_y >= bar2_y - 7.5 and circle_y <= bar2_y + 42.5:
                    circle_x = 605.
                    speed_x = -speed_x
            if circle_x < 5.:
                bar2_score += 1
                circle_x, circle_y = 320., 232.5
                bar1_y,bar_2_y = 215., 215.
            elif circle_x > 620.:
                bar1_score += 1
                circle_x, circle_y = 307.5, 232.5
                bar1_y, bar2_y = 215., 215.
            if circle_y <= 10.:
                speed_y = -speed_y
                circle_y = 10.
            elif circle_y >= 457.5:
                speed_y = -speed_y
                circle_y = 457.5

            if e_time - s_time >= 60:
                is_playing = False
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False
            
            pygame.display.flip()
        cv2.destroyAllWindows()
        self.score = str(bar1_score) + " : " + str(bar2_score)

        
        


if __name__ == "__main__":
    gui = GUI()