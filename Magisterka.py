# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import mediapipe as mp
import cv2
import math
from keras.models import load_model
import numpy as np

def info(image):
    fingerCount = 0
    len1 = 1
    len2 = 1
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    angle = 0
    direction = ''

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label

            handLandmarks = []

            for landmarks in hand_landmarks.landmark:
                handLandmarks.append([landmarks.x, landmarks.y])

            if handLabel == "Left":
                if handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCount = fingerCount + 1
                if handLandmarks[8][1] < handLandmarks[6][1]:
                    fingerCount = fingerCount + 1
                if handLandmarks[12][1] < handLandmarks[10][1]:
                    fingerCount = fingerCount + 1
                if handLandmarks[16][1] < handLandmarks[14][1]:
                    fingerCount = fingerCount + 1
                if handLandmarks[20][1] < handLandmarks[18][1]:
                    fingerCount = fingerCount + 1

            if handLabel == "Right":
                x1 = handLandmarks[5][0] * dimensions[1]
                y1 = handLandmarks[5][1] * dimensions[0]

                x2 = handLandmarks[8][0] * dimensions[1]
                y2 = handLandmarks[8][1] * dimensions[0]

                len1 = math.sqrt((x1 - x2) ** 2)
                len2 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                angle = round(math.asin(len1 / len2) * 180 / math.pi)

                if x1 < x2:
                    direction = 'Right'
                else:
                    direction = 'Left'

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    return angle, direction, fingerCount, x1, x2, y1, y2, image



def printInfo(image):

    cv2.putText(image, 'Speed: ' + str(speed) + '%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)
    cv2.putText(image, str(angle) + ' ' + direction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.line(image, (round(x2), round(y2)), (round(x1), round(y2)), (0, 255, 255), 2)
    cv2.line(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 255), 2)

    return image


if __name__ == '__main__':

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles

    mode = 'steering'
    holder = 'gesture'

    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            imageGesture = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

            dimensions = frame.shape

            if cv2.waitKey(5) == ord('k'):
                mode, holder = holder, mode

            if mode == 'steering':
                results = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            angle, direction, fingerCount, x1, x2, y1, y2, image = info(image)
            angle = round(angle/5) * 5
            speed = fingerCount * 20

            if mode == 'steering':
                image = printInfo(image)

            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                     circle_radius=1),
                                              mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=1, circle_radius=1),
                                              )


            imageGesture = np.asarray(imageGesture, dtype=np.float32).reshape(1, 224, 224, 3)
            imageGesture = (imageGesture / 127.5) - 1

            if mode == 'gesture':
                prediction = model.predict(imageGesture)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                gesture = class_name[2:]
                cv2.putText(image, 'Gesture: ' + str(gesture), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
                cv2.putText(image, 'Confidence: ' + str(np.round(confidence_score * 100))[:-2], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                            2)

            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()