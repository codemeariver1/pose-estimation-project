import cv2
import PoseModule as pm
import time

capture = cv2.VideoCapture(0)
prevTime = 0
detector = pm.PoseDetector()

while True:
    success, img = capture.read()
    img = detector.findPose(img)
    landmark_list = detector.findPosition(img, draw=False)
    if len(landmark_list) != 0:
        print(landmark_list[11])
        cv2.circle(img, (landmark_list[11][1], landmark_list[11][2]), 15, (0, 0, 255), cv2.FILLED)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)