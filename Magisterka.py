# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import mediapipe as mp
import cv2
import numpy as np
import uuid

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

            image.flags.writeable = False

            results = hands.process(image)

            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            print(results)

            fingerCount = 0

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

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            cv2.putText(image, 'predkosc: ' + str(fingerCount * 20) + '%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

            # narysowanie zidentyfikowanych punktow charakterystycznych dloni:
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                     circle_radius=1),
                                              mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=1, circle_radius=1),
                                              )

            cv2.imshow('Hand Tracking', image)

            # zapisywanie do folderu:
            #          cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
            #          cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
