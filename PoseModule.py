import cv2
import mediapipe as mp
import time


class PoseDetector():
    def __init__(
            self, mode=False, smooth=True, complexity=1, segmentation=False,
            smooth_segmentation=False, detection_conf=0.5, tracking_conf=0.5):
        self.mode = mode
        self.smooth = smooth
        self.complexity = complexity
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode, self.smooth, self.complexity, self.segmentation,
            self.smooth_segmentation, self.detection_conf, self.tracking_conf
        )

    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        landmark_list = []
        if self.results.pose_landmarks:
            for bodyId, landmark in enumerate(self.results.pose_landmarks.landmark):
                    h, w, ch = img.shape
                    # print(bodyId, landmark)
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append([bodyId, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return landmark_list


def main():
    capture = cv2.VideoCapture(0)
    prevTime = 0
    detector = PoseDetector()

    while True:
        success, img = capture.read()
        img = detector.findPose(img)
        landmark_list = detector.findPosition(img, draw=False)
        if len(landmark_list) != 0:
            print(landmark_list[14])
            cv2.circle(img, (landmark_list[14][1], landmark_list[14][2]), 15, (0, 0, 255), cv2.FILLED)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()