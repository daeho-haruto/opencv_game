import cv2
import mediapipe as mp
import numpy as np
import time

count = [0,0,0,0,0]

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

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

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
                # cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': rps_gesture[idx],
                    'org': org
                })

            # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            try:
                if rps_result[0]['rps'] == 'cancel':
                    count = [0,0,0,0,0]
                    cv2.putText(img, "cancel", (10,30), 0, 1, (0, 0, 255), 2)
                elif rps_result[0]['rps'] == 'one':
                    count[0] += 1
                    cv2.putText(img, "drop object Game", (10,30), 0, 1, (0, 0, 255), 2)
                    if count[0] == 50: 
                        cv2.destroyWindow('Game')
                        import dropLogo
                elif rps_result[0]['rps'] == 'two':
                    count[1] += 1
                    cv2.putText(img, "catch object Game", (10,30), 0, 1, (0, 0, 255), 2)
                    if count[1] == 50:
                        import logo_Catch
                elif rps_result[0]['rps'] == 'three':
                    count[2] += 1
                    cv2.putText(img, "ping-pong Game", (10,30), 0, 1, (0, 0, 255), 2)
                    if count[2] == 50: 
                        import pong
                elif rps_result[0]['rps'] == 'four':
                    count[3] += 1
                    cv2.putText(img, "snake Game", (10,30), 0, 1, (0, 0, 255), 2)
                    if count[3] == 50: exit()
                elif rps_result[0]['rps'] == 'five':
                    count[4] += 1
                    cv2.putText(img, "Pac-Man Game", (10,30), 0, 1, (0, 0, 255), 2)
                    if count[4] == 50: exit()

            except IndexError:
                pass
            
    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()