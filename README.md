# Masters thesis
Goal of my masters thesis is to implement algorithm that would allow user to control mobile platform using hand gestures. 
To achieve my goal i'll be basing on Mediapipe hand gestures model that detects 21 hand landmarks.
## Required modules:
* OpenCV
* Mediapipe
* Pygame
* Math
* Time
## Installation guide:
To ensure that your project will work fine first you'll need to install necessary modules. You can do it simply by running following commands in your terminal:  
pip install opencv-python  
pip install mediapipe  
pip install pygame  
## Setup
First thing to do is initiating opencv and mediapipe facial recognition model
```
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
```
Then you'll need to setup your model
```
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
```
Next you'll want to start your webcam and process image by mediapipe model
```
        while cap.isOpened():
            success, image = cap.read()
            results = face_mesh.process(image)
```
What you get as a result of processing your image is list containing 468 landmarks, each having their x,y and z position in image.
For analysing your eye you'll need 8 landmarks seen as below:  
![eyes](https://user-images.githubusercontent.com/130605144/232501991-340835ef-d372-483a-ba7f-44d2cdd64f48.png)  
Points indices: P1 = 385, P2 = 387, P3 = 380, P4 = 373, P5 = 160, P6 = 158, P7 = 144, P8 = 153  
To detect whether eyes are opened or closed you can calculate the distance between top and bottom pair of points. To do so you can simply calculate euclidean distance: 
```
    def distance(p1, p2):
        x1 = p1
        x2 = p2
        dist = math.sqrt((x1 - x2)**2)
        return dist
```
Next step in your project is extracting x and y values from results list:
```
            leftEye = [[results.multi_face_landmarks[0].landmark[385].x * imgW, results.multi_face_landmarks[0].landmark[385].y * imgH],
                       [results.multi_face_landmarks[0].landmark[387].x * imgW, results.multi_face_landmarks[0].landmark[387].y * imgH],
                       [results.multi_face_landmarks[0].landmark[380].x * imgW, results.multi_face_landmarks[0].landmark[380].y * imgH],
                       [results.multi_face_landmarks[0].landmark[373].x * imgW, results.multi_face_landmarks[0].landmark[373].y * imgH]]

            rightEye = [[results.multi_face_landmarks[0].landmark[160].x * imgW, results.multi_face_landmarks[0].landmark[160].y * imgH],
                       [results.multi_face_landmarks[0].landmark[158].x * imgW, results.multi_face_landmarks[0].landmark[158].y * imgH],
                       [results.multi_face_landmarks[0].landmark[144].x * imgW, results.multi_face_landmarks[0].landmark[144].y * imgH],
                       [results.multi_face_landmarks[0].landmark[153].x * imgW, results.multi_face_landmarks[0].landmark[153].y * imgH],
```
To detect whether eye is opened or not you'll only need the y variable but for debugging purpose x variable was also extracted.  
Once you have your y variable extracted you can calculate euclidean distance:
```
            dist1 = distance(leftEye[0][1], leftEye[2][1])
            dist2 = distance(leftEye[1][1], leftEye[3][1])

            dist3 = distance(rightEye[0][1], rightEye[2][1])
            dist4 = distance(rightEye[1][1], rightEye[3][1])
```
Now that you have calculated distances between points we can add variables that'll store state of your eye:
```
            if dist1 and dist2 < 1.5:
                leftEyeState = 'closed'
            else:
                leftEyeState = 'opened'

            if dist3 and dist4 < 1.5:
                rightEyeState = 'closed'
            else:
                rightEyeState = 'opened'
```
1.5 which is threshold value between closed and opened eye was set with trial and error method so it may vary for different people.  
Next step in your app would be to calculate how long your eyes are closed. You can do that by using time module:
```
            while(leftEyeState == 'opened' and rightEyeState == 'opened'):
                t1 = time.time()
                t2 = 0
                break

            while(leftEyeState == 'closed' and rightEyeState == 'closed'):
                t2 = time.time()
                break
            czas = t2 - t1
```
Second to last step is to initialize pygame method called mixer which'll allow you to play alarm:
```
            mixer.init()
            sound = mixer.Sound('alarm.wav')
```
Last step of your project is to play your alarm when eyes are being closed for too long. In my case i chose value of 1.5 seconds:
```
            while(czas > 1.5):
                sound.play()
                break
```
What you can also do is display your opencv view and print in console state of your eyes:
```
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

            print(f'Left eye is {leftEyeState}, Right eye is {rightEyeState}')
```
Once you have everything above setup, you have to add last commands that'll tell opencv when to close your app:
```
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
```
