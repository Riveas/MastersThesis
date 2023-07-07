# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import mediapipe as mp
import cv2
import math
import numpy as np
import uuid

def slope(x1, x2, y1, y2):
    if x1 == x2:
        x1 = x1 + 1
    return (y2-y1)/(x1-x2)


if __name__ == '__main__':

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            dimensions = frame.shape

            results = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            fingerCount = 0
            finLen = 0
            x1 = 0
            x2 = 0
            y1 = 0
            y2 = 0

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                    handLabel = results.multi_handedness[handIndex].classification[0].label

                    handLandmarks = []

                    for landmarks in hand_landmarks.landmark:
                        handLandmarks.append([landmarks.x, landmarks.y])

                    if handLabel == "Right":
                        if handLandmarks[4][0] < handLandmarks[3][0]:
                            fingerCount = fingerCount + 1
                        if handLandmarks[8][1] < handLandmarks[6][1]:
                            fingerCount = fingerCount + 1
                        if handLandmarks[12][1] < handLandmarks[10][1]:
                            fingerCount = fingerCount + 1
                        if handLandmarks[16][1] < handLandmarks[14][1]:
                            fingerCount = fingerCount + 1
                        if handLandmarks[20][1] < handLandmarks[18][1]:
                            fingerCount = fingerCount + 1

                    if handLabel == "Left":
                        x1 = handLandmarks[5][0]*dimensions[1]
                        y1 = handLandmarks[5][1]*dimensions[0]

                        x2 = handLandmarks[8][0]*dimensions[1]
                        y2 = handLandmarks[8][1]*dimensions[0]

                    finLen = math.sqrt((x2-x1)**2 + (y2-y1)**2)


                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            line1 = ((math.floor(x1), math.floor(y1)), (math.floor(x2), math.floor(y2)))
            line2 = ((math.floor(x1), math.floor(y2)), (math.floor(x2), math.floor(y2)))

            slope1 = slope(line1[0][0], line1[0][1], line1[1][0], line1[1][1])
            slope2 = slope(line2[0][0], line2[0][1], line2[1][0], line2[1][1])
            angle = abs(math.degrees(math.atan((slope2-slope1)/(1+(slope2*slope1)))))

            cv2.putText(image, 'predkosc: ' + str(fingerCount * 20) + '%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)
            cv2.putText(image, str(angle), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.line(image, (round(x2), round(y2)), (round(x1), round(y2)), (0, 255, 255), 2)
            cv2.line(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 255), 2)

            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                     circle_radius=1),
                                              mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=1, circle_radius=1),
                                              )

            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
