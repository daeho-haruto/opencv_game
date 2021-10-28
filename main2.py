import cv2
import pygame
import mediapipe as mp
import numpy as np
import time
import pyautogui
import random

max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok'
}
rps_gesture = {0:'cancel', 1:'one', 9:'two', 3:'three', 4:'four', 5:'five'}

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
    rank = [[],[],[]]
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
                    self.screen.blit(pygame.font.SysFont('None', 72).render({'cancel': 'cancel', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5'}[state], True, (0, 255, 0)), (0, 0))
                    if state == last_state:
                        state_cnt += 1
                    else:
                        state_cnt = 0
                    if state_cnt >= 50:
                        if state == 'cancel':
                            if state_cnt == 70:
                                gameEnd = True
                                state_cnt = 0
                        elif state == 'one':
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
                            self.pacman()
                            state_cnt = 0
                        elif state == 'five':
                            self.showRank()
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
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(self.score), True, (0, 255, 0)), (0, 0))
            if e_time - s_time >= 5:
                break
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_showing = False
            pygame.display.flip()

    def dropLogo(self):
        s_time = time.time()
        is_playing = True
        while is_playing:
            _, bg_frame = cap.read()
            bg_frame = cv2.resize(bg_frame,self.size)
            bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
            bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
            
            e_time = time.time()

            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(bg_frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(5-int(e_time-s_time)), True, (0, 255, 0)), (0, 0))
            
            if e_time - s_time >= 5 :
                break
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False

            pygame.display.flip()
    
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
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(30-int(end_time-start_time)),True, (0, 255, 0)), (580, 0))

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
            _, bg_frame = cap.read()
            bg_frame = cv2.resize(bg_frame,self.size)
            bg_frame = cv2.cvtColor(bg_frame,cv2.COLOR_BGR2RGB)
            e_time = time.time()
            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(bg_frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(5-int(e_time-s_time)), True, (0, 255, 0)), (0, 0))
            if e_time-s_time >= 5:
                is_playing = False

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    is_playing = False

            pygame.display.flip()

        class Object:
            def __init__(self, size=50):
                self.logo_org = cv2.imread('game\img\logo.png')
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
                num_x = random.randint(0,590)
                num_y = random.randint(0,430)

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
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(30-last_time), True, (0, 255, 0)), (580, 0))

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
            _, bg_frame = cap.read()
            bg_frame = cv2.resize(bg_frame,self.size)
            bg_frame = cv2.cvtColor(bg_frame,cv2.COLOR_BGR2RGB)
            e_time = time.time()
            self.screen.blit(pygame.transform.rotate(pygame.surfarray.make_surface(bg_frame), 270), (0, 0))
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(5-int(e_time-s_time)), True, (0, 255, 0)), (0, 0))
            if e_time-s_time >= 5:
                is_playing = False

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
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
                    if state == 'cancel':
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
            self.screen.blit(pygame.font.SysFont('None', 72).render(str(60-int(e_time-s_time)), True, (0, 255, 0)), (580, 0))

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
    
    def pacman(self):
        black = (0,0,0)
        white = (255,255,255)
        blue = (0,0,255)
        green = (0,255,0)
        red = (255,0,0)
        purple = (255,0,255)
        yellow   = ( 255, 255, 0)
        Trollicon=pygame.image.load('game\Pacman\images\Trollman.png')
        pygame.display.set_icon(Trollicon)

        pygame.mixer.init()
        pygame.mixer.music.load('game\Pacman\pacman.mp3')
        pygame.mixer.music.play(-1, 0.0)

        class Wall(pygame.sprite.Sprite):
        # Constructor function
            def __init__(self,x,y,width,height, color):
                # Call the parent's constructor
                pygame.sprite.Sprite.__init__(self)
        
                # Make a blue wall, of the size specified in the parameters
                self.image = pygame.Surface([width, height])
                self.image.fill(color)
        
                # Make our top-left corner the passed-in location.
                self.rect = self.image.get_rect()
                self.rect.top = y
                self.rect.left = x
            
        def setupRoomOne(all_sprites_list):
        # Make the walls. (x_pos, y_pos, width, height)
            wall_list=pygame.sprite.RenderPlain()
            
            # This is a list of walls. Each is in the form [x, y, width, height]
            walls = [ [0,0,6,600],
                    [0,0,600,6],
                    [0,600,606,6],
                    [600,0,6,606],
                    [300,0,6,66],
                    [60,60,186,6],
                    [360,60,186,6],
                    [60,120,66,6],
                    [60,120,6,126],
                    [180,120,246,6],
                    [300,120,6,66],
                    [480,120,66,6],
                    [540,120,6,126],
                    [120,180,126,6],
                    [120,180,6,126],
                    [360,180,126,6],
                    [480,180,6,126],
                    [180,240,6,126],
                    [180,360,246,6],
                    [420,240,6,126],
                    [240,240,42,6],
                    [324,240,42,6],
                    [240,240,6,66],
                    [240,300,126,6],
                    [360,240,6,66],
                    [0,300,66,6],
                    [540,300,66,6],
                    [60,360,66,6],
                    [60,360,6,186],
                    [480,360,66,6],
                    [540,360,6,186],
                    [120,420,366,6],
                    [120,420,6,66],
                    [480,420,6,66],
                    [180,480,246,6],
                    [300,480,6,66],
                    [120,540,126,6],
                    [360,540,126,6]
                    ]
            
            # Loop through the list. Create the wall, add it to the list
            for item in walls:
                wall=Wall(item[0],item[1],item[2],item[3],blue)
                wall_list.add(wall)
                all_sprites_list.add(wall)
                
            # return our new list
            return wall_list
        def setupGate(all_sprites_list):
            gate = pygame.sprite.RenderPlain()
            gate.add(Wall(282,242,42,2,white))
            all_sprites_list.add(gate)
            return gate

        class Block(pygame.sprite.Sprite):
            # Constructor. Pass in the color of the block, 
            # and its x and y position
            def __init__(self, color, width, height):
                # Call the parent class (Sprite) constructor
                pygame.sprite.Sprite.__init__(self) 
        
                # Create an image of the block, and fill it with a color.
                # This could also be an image loaded from the disk.
                self.image = pygame.Surface([width, height])
                self.image.fill(white)
                self.image.set_colorkey(white)
                pygame.draw.ellipse(self.image,color,[0,0,width,height])
        
                # Fetch the rectangle object that has the dimensions of the image
                # image.
                # Update the position of this object by setting the values 
                # of rect.x and rect.y
                self.rect = self.image.get_rect() 

        class Player(pygame.sprite.Sprite):
            # Set speed vector
            change_x=0
            change_y=0
        
            # Constructor function
            def __init__(self,x,y, filename):
                # Call the parent's constructor
                pygame.sprite.Sprite.__init__(self)
        
                # Set height, width
                self.image = pygame.image.load(filename).convert()
        
                # Make our top-left corner the passed-in location.
                self.rect = self.image.get_rect()
                self.rect.top = y
                self.rect.left = x
                self.prev_x = x
                self.prev_y = y

            def prevdirection(self):
                self.prev_x = self.change_x
                self.prev_y = self.change_y
            # Change the speed of the player
            def changespeed(self,x,y):
                self.change_x+=x
                self.change_y+=y
            def update(self,walls,gate):
            # Get the old position, in case we need to go back to it
                old_x=self.rect.left
                new_x=old_x+self.change_x
                prev_x=old_x+self.prev_x
                self.rect.left = new_x
                
                old_y=self.rect.top
                new_y=old_y+self.change_y
                prev_y=old_y+self.prev_y

                # Did this update cause us to hit a wall?
                x_collide = pygame.sprite.spritecollide(self, walls, False)
                if x_collide:
                    # Whoops, hit a wall. Go back to the old position
                    self.rect.left=old_x
                    # self.rect.top=prev_y
                    # y_collide = pygame.sprite.spritecollide(self, walls, False)
                    # if y_collide:
                    #     # Whoops, hit a wall. Go back to the old position
                    #     self.rect.top=old_y
                    #     print('a')
                else:
                    self.rect.top = new_y
                    # Did this update cause us to hit a wall?
                    y_collide = pygame.sprite.spritecollide(self, walls, False)
                    if y_collide:
                        # Whoops, hit a wall. Go back to the old position
                        self.rect.top=old_y
                        # self.rect.left=prev_x
                        # x_collide = pygame.sprite.spritecollide(self, walls, False)
                        # if x_collide:
                        #     # Whoops, hit a wall. Go back to the old position
                        #     self.rect.left=old_x
                        #     print('b')
                if gate != False:
                    gate_hit = pygame.sprite.spritecollide(self, gate, False)
                    if gate_hit:
                        self.rect.left=old_x
                        self.rect.top=old_y
        class Ghost(Player):
            # Change the speed of the ghost
            def changespeed(self,list,ghost,turn,steps,l):
                try:
                    z=list[turn][2]
                    if steps < z:
                        self.change_x=list[turn][0]
                        self.change_y=list[turn][1]
                        steps+=1
                    else:
                        if turn < l:
                            turn+=1
                        elif ghost == "clyde":
                            turn = 2
                        else:
                            turn = 0
                        self.change_x=list[turn][0]
                        self.change_y=list[turn][1]
                        steps = 0
                    return [turn,steps]
                except IndexError:
                    return [0,0]
        Pinky_directions = [
        [0,-30,4],
        [15,0,9],
        [0,15,11],
        [-15,0,23],
        [0,15,7],
        [15,0,3],
        [0,-15,3],
        [15,0,19],
        [0,15,3],
        [15,0,3],
        [0,15,3],
        [15,0,3],
        [0,-15,15],
        [-15,0,7],
        [0,15,3],
        [-15,0,19],
        [0,-15,11],
        [15,0,9]
        ]

        Blinky_directions = [
        [0,-15,4],
        [15,0,9],
        [0,15,11],
        [15,0,3],
        [0,15,7],
        [-15,0,11],
        [0,15,3],
        [15,0,15],
        [0,-15,15],
        [15,0,3],
        [0,-15,11],
        [-15,0,3],
        [0,-15,11],
        [-15,0,3],
        [0,-15,3],
        [-15,0,7],
        [0,-15,3],
        [15,0,15],
        [0,15,15],
        [-15,0,3],
        [0,15,3],
        [-15,0,3],
        [0,-15,7],
        [-15,0,3],
        [0,15,7],
        [-15,0,11],
        [0,-15,7],
        [15,0,5]
        ]

        Inky_directions = [
        [30,0,2],
        [0,-15,4],
        [15,0,10],
        [0,15,7],
        [15,0,3],
        [0,-15,3],
        [15,0,3],
        [0,-15,15],
        [-15,0,15],
        [0,15,3],
        [15,0,15],
        [0,15,11],
        [-15,0,3],
        [0,-15,7],
        [-15,0,11],
        [0,15,3],
        [-15,0,11],
        [0,15,7],
        [-15,0,3],
        [0,-15,3],
        [-15,0,3],
        [0,-15,15],
        [15,0,15],
        [0,15,3],
        [-15,0,15],
        [0,15,11],
        [15,0,3],
        [0,-15,11],
        [15,0,11],
        [0,15,3],
        [15,0,1],
        ]

        Clyde_directions = [
        [-30,0,2],
        [0,-15,4],
        [15,0,5],
        [0,15,7],
        [-15,0,11],
        [0,-15,7],
        [-15,0,3],
        [0,15,7],
        [-15,0,7],
        [0,15,15],
        [15,0,15],
        [0,-15,3],
        [-15,0,11],
        [0,-15,7],
        [15,0,3],
        [0,-15,11],
        [15,0,9],
        ]

        pl = len(Pinky_directions)-1
        bl = len(Blinky_directions)-1
        il = len(Inky_directions)-1
        cl = len(Clyde_directions)-1

        pygame.init()

        self.screen = pygame.display.set_mode([606, 606])

        pygame.display.set_caption('Pacman')

        # Create a surface we can draw on
        background = pygame.Surface(self.screen.get_size())

        # Used for converting color maps and such
        background = background.convert()
        
        # Fill the screen with a black background
        background.fill(black)

        clock = pygame.time.Clock()

        pygame.font.init()
        font = pygame.font.Font("game\Pacman\dfreesansbold.ttf", 24)

        w = 303-16 #Width
        p_h = (7*60)+19 #Pacman height
        m_h = (4*60)+19 #Monster height
        b_h = (3*60)+19 #Binky height
        i_w = 303-16-32 #Inky width
        c_w = 303+(32-16) #Clyde width

        def startGame():
            all_sprites_list = pygame.sprite.RenderPlain()

            block_list = pygame.sprite.RenderPlain()

            monsta_list = pygame.sprite.RenderPlain()

            pacman_collide = pygame.sprite.RenderPlain()

            wall_list = setupRoomOne(all_sprites_list)

            gate = setupGate(all_sprites_list)


            p_turn = 0
            p_steps = 0

            b_turn = 0
            b_steps = 0

            i_turn = 0
            i_steps = 0

            c_turn = 0
            c_steps = 0


            # Create the player paddle object
            Pacman = Player( w, p_h, "game\Pacman\images\Trollman.png" )
            all_sprites_list.add(Pacman)
            pacman_collide.add(Pacman)
            
            Blinky=Ghost( w, b_h, "game\Pacman\images\Blinky.png" )
            monsta_list.add(Blinky)
            all_sprites_list.add(Blinky)

            Pinky=Ghost( w, m_h, "game\Pacman\images\Pinky.png" )
            monsta_list.add(Pinky)
            all_sprites_list.add(Pinky)
            
            Inky=Ghost( i_w, m_h, "game\Pacman\images\Inky.png" )
            monsta_list.add(Inky)
            all_sprites_list.add(Inky)
            
            Clyde=Ghost( c_w, m_h, "game\Pacman\images\Clyde.png" )
            monsta_list.add(Clyde)
            all_sprites_list.add(Clyde)

            for row in range(19):
                for column in range(19):
                    if (row == 7 or row == 8) and (column == 8 or column == 9 or column == 10):
                        continue
                    else:
                        block = Block(yellow, 4, 4)
                        # Set a random location for the block
                        block.rect.x = (30*column+6)+26
                        block.rect.y = (30*row+6)+26

                        b_collide = pygame.sprite.spritecollide(block, wall_list, False)
                        p_collide = pygame.sprite.spritecollide(block, pacman_collide, False)
                        if b_collide:
                            continue
                        elif p_collide:
                            continue
                        else:
                            # Add the block to the list of objects
                            block_list.add(block)
                            all_sprites_list.add(block)

            bll = len(block_list)

            score = 0

            done = False

            i = 0

            while done == False:
            # ALL EVENT PROCESSING SHOULD GO BELOW THIS COMMENT
                Pacman.changespeed()
                # for event in pygame.event.get():
                #     if event.type == pygame.QUIT:
                #         done=True

                #     if event.type == pygame.KEYDOWN:
                #         if event.key == pygame.K_LEFT:
                #             Pacman.changespeed(-30,0)
                #         if event.key == pygame.K_RIGHT:
                #             Pacman.changespeed(30,0)
                #         if event.key == pygame.K_UP:
                #             Pacman.changespeed(0,-30)
                #         if event.key == pygame.K_DOWN:
                #             Pacman.changespeed(0,30)

                #     if event.type == pygame.KEYUP:
                #         if event.key == pygame.K_LEFT:
                #             Pacman.changespeed(30,0)
                #         if event.key == pygame.K_RIGHT:
                #             Pacman.changespeed(-30,0)
                #         if event.key == pygame.K_UP:
                #             Pacman.changespeed(0,30)
                #         if event.key == pygame.K_DOWN:
                #             Pacman.changespeed(0,-30)
            
                Pacman.update(wall_list,gate)

                returned = Pinky.changespeed(Pinky_directions,False,p_turn,p_steps,pl)
                p_turn = returned[0]
                p_steps = returned[1]
                Pinky.changespeed(Pinky_directions,False,p_turn,p_steps,pl)
                Pinky.update(wall_list,False)

                returned = Blinky.changespeed(Blinky_directions,False,b_turn,b_steps,bl)
                b_turn = returned[0]
                b_steps = returned[1]
                Blinky.changespeed(Blinky_directions,False,b_turn,b_steps,bl)
                Blinky.update(wall_list,False)

                returned = Inky.changespeed(Inky_directions,False,i_turn,i_steps,il)
                i_turn = returned[0]
                i_steps = returned[1]
                Inky.changespeed(Inky_directions,False,i_turn,i_steps,il)
                Inky.update(wall_list,False)

                returned = Clyde.changespeed(Clyde_directions,"clyde",c_turn,c_steps,cl)
                c_turn = returned[0]
                c_steps = returned[1]
                Clyde.changespeed(Clyde_directions,"clyde",c_turn,c_steps,cl)
                Clyde.update(wall_list,False)

                # See if the Pacman block has collided with anything.
                blocks_hit_list = pygame.sprite.spritecollide(Pacman, block_list, True)

                if len(blocks_hit_list) > 0:
                    score +=len(blocks_hit_list)
                
                self.screen.fill(black)
        
                wall_list.draw(self.screen)
                gate.draw(self.screen)
                all_sprites_list.draw(self.screen)
                monsta_list.draw(self.screen)

                text=font.render("Score: "+str(score)+"/"+str(bll), True, red)
                self.screen.blit(text, [10, 10])

                if score == bll:
                    doNext("Congratulations, you won!",145,all_sprites_list,block_list,monsta_list,pacman_collide,wall_list,gate)

                monsta_hit_list = pygame.sprite.spritecollide(Pacman, monsta_list, False)

                if monsta_hit_list:
                    doNext("Game Over",235,all_sprites_list,block_list,monsta_list,pacman_collide,wall_list,gate)

                pygame.display.flip()

                clock.tick(10)
        def doNext(message,left,all_sprites_list,block_list,monsta_list,pacman_collide,wall_list,gate):
            while True:
                # ALL EVENT PROCESSING SHOULD GO BELOW THIS COMMENT
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                        if event.key == pygame.K_RETURN:
                            del all_sprites_list
                            del block_list
                            del monsta_list
                            del pacman_collide
                            del wall_list
                            del gate
                            startGame()
                
                w = pygame.Surface((400,200))  # the size of your rect
                w.set_alpha(10)                # alpha level
                w.fill((128,128,128))           # this fills the entire surface
                self.screen.blit(w, (100,200))    # (0,0) are the top-left coordinates

                #Won or lost
                text1=font.render(message, True, white)
                self.screen.blit(text1, [left, 233])

                text2=font.render("To play again, press ENTER.", True, white)
                self.screen.blit(text2, [135, 303])
                text3=font.render("To quit, press ESCAPE.", True, white)
                self.screen.blit(text3, [165, 333])

                pygame.display.flip()

                clock.tick(10)
        def opencv():
            PINK_MIN = np.array([100,80,2], np.uint8)
            PINK_MAX = np.array([126,255,255], np.uint8)

            centroid_x = 0
            centroid_y = 0
            s = ''
            move = ''

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


        is_playing = True
        while is_playing :
            opencv()
            startGame()
            
            

            



if __name__ == "__main__":
    gui = GUI()
