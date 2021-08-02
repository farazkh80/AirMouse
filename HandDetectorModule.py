"""
A hand detection and finger tracking module built with mediapipe solutions
and openCV for finger coordinate detection and status tracking
"""

import cv2
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(self, mode=False, maxHands = 1, detectionCon=0.5, trackCon=0.5) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.lmList = []
        self.tipIds = [4,8,12,16,20]
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def findLandmarkPosition(self, img, handNo=0, draw=True) -> list:
        self.lmList = []

        if self.results.multi_hand_landmarks:
            detectedHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(detectedHand.landmark):
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                    
                self.lmList.append([id, x, y])
               

        return self.lmList

    def drawCircleOnPoint(self, img, pId) -> None:
        self.findLandmarkPosition(img)
        if len(self.lmList) != 0 and pId >=0 and pId <= 20:
            cv2.circle(img, (self.lmList[pId][1], self.lmList[pId][2]), 10, (255, 0, 255), cv2.FILLED)

    def multipleFingersUp(self) -> list:
        fingers = []
        if len(self.lmList) != 0:
            # Thumb
            # facing inside of hand
            if self.lmList[1][1] < self.lmList[0][1]:
                if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # facing outside of hand
            else:
                if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Fingers
            for id in range(1, 5):

                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def fingerUp(self, tipId) -> bool:
        if len(self.lmList) != 0 and tipId >=0 and tipId <= 20 and tipId % 4 == 0:
            # Thumb
            if(tipId == 4):
                if self.lmList[1][1] < self.lmList[0][1]:
                    if self.lmList[tipId][1] < self.lmList[tipId - 2][1]:
                        return True
                    else:
                        return False
                else:
                    if self.lmList[tipId][1] > self.lmList[tipId - 2][1]:
                        return True
                    else:
                        return False

            else:
                if self.lmList[tipId][2] < self.lmList[tipId - 2][2]:
                    return True
                else:
                    return False

        return False

    def getFingerUpCoords(self, tipId):
        x, y = None, None
        if(self.fingerUp(tipId)):
            x, y = self.lmList[tipId][1], self.lmList[tipId][2]

        return (x, y)

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        detector.findHands(img)

        detector.findLandmarkPosition(img, 0)
        fingers = detector.multipleFingersUp()

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (2555,0,255), 2)
        cv2.imshow("Live Capture", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()