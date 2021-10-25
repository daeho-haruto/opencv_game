import pygame
from pygame.locals import *
import cv2
import numpy as np
import mediapipe as mp

max_num_hands = 1
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}
rps_gesture = {0:'down', 5:'up'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


file = np.genfromtxt('game\data\gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(1)

def opencv():
    ret, img = cap.read()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        rps_result = []
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

          
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
        
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle)

         
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

           
            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                rps_result.append({
                    'rps': rps_gesture[idx],
                    'org': org
                })

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            try:
                if rps_result[0]['rps'] == 'down':
                    return 'down'
                elif rps_result[0]['rps'] == 'up':
                    return 'up'
            except IndexError: 
                pass

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

pygame.init()

screen=pygame.display.set_mode((640,480),0,32)
pygame.display.set_caption("Pong Pong!")


back = pygame.Surface((640,480))
background = back.convert()
background.fill((0,0,0))
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

while True:
    position = opencv()
    
    if position == 'down':
        bar1_move = ai_speed
    elif position == 'up':
        bar1_move = -ai_speed
    
    
    # for event in pygame.event.get():
    #     if event.type == QUIT:
    #         exit()
    #     if event.type == KEYDOWN:
    #         if event.key == K_UP:
    #             bar1_move = -ai_speed
    #         elif event.key == K_DOWN:
    #             bar1_move = ai_speed
    #     elif event.type == KEYUP:
    #         if event.key == K_UP:
    #             bar1_move = 0.
    #         elif event.key == K_DOWN:
    #             bar1_move = 0.
    
    score1 = font.render(str(bar1_score), True,(255,255,255))
    score2 = font.render(str(bar2_score), True,(255,255,255))

    screen.blit(background,(0,0))
    frame = pygame.draw.rect(screen,(255,255,255),Rect((5,5),(630,470)),2)
    middle_line = pygame.draw.aaline(screen,(255,255,255),(330,5),(330,475))
    screen.blit(bar1,(bar1_x,bar1_y))
    screen.blit(bar2,(bar2_x,bar2_y))
    screen.blit(circle,(circle_x,circle_y))
    screen.blit(score1,(250.,210.))
    screen.blit(score2,(380.,210.))

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

    pygame.display.update()

    
cv2.destroyAllWindows()