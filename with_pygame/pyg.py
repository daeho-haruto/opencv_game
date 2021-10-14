import pygame
import cv2
import mediapipe as mp
import numpy as np
import sys

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

# Gesture recognition model
file = np.genfromtxt('with_pygame\gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)


black = (0,0,0)
white = (255,255,255)

pygame.init()

size = 700,500
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Ping Pong")

done = False
clock = pygame.time.Clock()

def player1(x1, y1, xsize, ysize):
    pygame.draw.rect(screen, black, [x1, y1, xsize, ysize])

def player2(x2, y2, xsize, ysize):
    pygame.draw.rect(screen, black, [x2,y2,xsize,ysize])

def ball(ballx, bally):
    pygame.draw.circle(screen, black, [ballx,bally],20)

def Score1(score1):
    font = pygame.font.Font(None ,50)
    text = font.render(str(score1), True, black)
    screen.blit(text, [160, 0])

def Score2(score2):
    font = pygame.font.Font(None ,50)
    text = font.render(str(score2), True, black)
    screen.blit(text, [510, 0])

def opencv():
    ret, img = cap.read()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
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
               cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]),
               int(res.landmark[0].y * img.shape[0]+20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
               fontScale=1, color=(255,255,255), thickness=1)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # Who wins?
    cv2.imshow('Game', img)

x1 = 5
y1 = 175
xsize = 35
ysize = 150
speed1 = 0

x2 = 660
y2 = 175
speed2 = 0

ballx = 350
bally = 250
speedx = 5
speedy = 5

score1 = 0
score2 = 0


while not done:
    opencv()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                speed1 = -10
            if event.key == pygame.K_s:
                speed1 = 10
            if event.key == pygame.K_UP:
                speed2 = -10
            if event.key == pygame.K_DOWN:
                speed2 = 10

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                speed1 = 0
            if event.key == pygame.K_s:
                speed1 = 0
            if event.key == pygame.K_UP:
                speed2 = 0
            if event.key ==  pygame.K_DOWN:
                speed2 = 0
                

    screen.fill(white)
    player1(x1, y1, xsize, ysize)
    player2(x2, y2, xsize, ysize)
    ball(ballx,bally)
    Score1(score1)
    Score2(score2)

    y1 += speed1
    y2 += speed2
    ballx += speedx
    bally += speedy

    if y1 < 0:
        y1 = 0

    if y1 > 350:
        y1 = 350

    if y2 < 0:
        y2 = 0

    if y2 > 350:
        y2 = 350

    if ballx+20 > x2 and bally-20 > y2 and bally+20 < y2+ysize and ballx < x2+3:
        speedx = -speedx

    if ballx-20 < x1+35 and bally-20 > y1 and bally+20 < y1+ysize and ballx > x1+38:
        speedx = -speedx

    if bally > 477 or bally < 23:
        speedy = -speedy

    if ballx < 13:
        score2 += 1
        ballx = 350
        bally = 250

    if ballx > 680:
        score1 += 1
        ballx = 350
        bally = 250
        

    pygame.display.flip()

    clock.tick(60)

pygame.quit()