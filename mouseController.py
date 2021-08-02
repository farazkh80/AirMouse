from modules.HandTrackModule import handDetector
from HandDetectorModule import HandDetector
import cv2
import numpy as np
import HandDetectorModule as hdm
from GestureDetector import GestureDetector
import autopy
import pyautogui
import pywhatkit

class MouseController():
    def __init__(self, pre_trained=False, gesture_dataset_file='hands-coords.csv', gesture_model_file=None, width=640, height=480):
        """
        Constructor
        :param pre_trained: whether the model will be uploaded from a file
        :param gesture_dataset_file: gesture dataset file name
        :param gesture_model_file: gesture model file name
        :param width: camera frame width
        :param height: camera frame height
        """
        self.cap = cv2.VideoCapture(0)
        self.autopy = autopy
        self.image = None
        
        self.wCam = width # camera frame width
        self.hCam =  height # camera frame height
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        self.wScr, self.hScr = self.autopy.screen.size() # screen width and height
        self.plocX, self.plocY = 0, 0 # previous x and y location of cursor
        self.clocX, self.clocY = 0, 0 # current x and y location of cursor
        self.frameR = 70 # frame Reduction
        self.smoothening = 7 # smoothing mouse move
        self.click_threshold = 40

        self.start = False # start airmouse
        self.gesture_enabled = False # Gesture mode or finger mode chooser

        self.handDetector =  hdm.HandDetector(maxHands=1)
        self.lmList = [] # list of landmark coordinates
        self.fingers = None # binary list of fingers indicating they are up or down
        # Finger tip ids
        self.thump = 4
        self.index = 8
        self.middle = 12
        self.ring = 16
        self.pinky = 20

        self.gesture_detector = GestureDetector(gesture_dataset_file)
        self.gesture = None # classified gesture
        self.gesture_prob = None # classified gesture probability
        self.pre_trained = pre_trained # pretrained model flag from a file
        self.gesture_model_file = gesture_model_file # pretrained model file name
        self.already_trained = False # traning complete flag

    def gesture_operation(self):
        """
        Automated operation handler based on the detected self.gesture
        """
        # Operation: zoom in
        if self.gesture == 'OPEN':
            pyautogui.keyDown('ctrl')
            pyautogui.keyDown('alt')
            pyautogui.scroll(50)
            pyautogui.keyUp('alt')
            pyautogui.keyUp('ctrl')

        # Operation: zoom out
        elif self.gesture == 'CLOSE':
            pyautogui.keyDown('ctrl')
            pyautogui.keyDown('alt')
            pyautogui.scroll(-50)
            pyautogui.keyUp('alt')
            pyautogui.keyUp('ctrl')

        # Operation: Play Video on youtube
        elif self.gesture == 'Thumps-Up':
            pywhatkit.playonyt("Country Songs")
            self.gesture_enabled = False

    def gesture_detection(self):
        """
        Detects the gesture class
        """
        # if the model is to be loaded from a file and is not already trained
        if self.pre_trained and self.gesture_model_file is not None and self.already_trained is False:
            self.gesture_detector.load_model_from_file(True, self.gesture_model_file)
        # if the model is to be trained and is not already trained
        elif self.already_trained is False:
            self.gesture_detector.train()
        
        self.already_trained = True

        # make detections from camera
        self.image = self.gesture_detector.make_detections_with_cv(self.cap, self.wCam, self.hCam)
        # get the detected gesture class and its probability
        self.gesture, self.gesture_prob = self.gesture_detector.get_detected_gesture()

        # call the gesture operation for automated operation handling
        self.gesture_operation()

    def fingers_operation(self):
        """
        Automated operation handler based on self.fingers status 0:down 1:up
        """
        if(len(self.lmList) != 0):
            # get fingers list indicating if each finger is up or down
            self.fingers = self.handDetector.multipleFingersUp()

            # draw a rectangle as a boundary
            cv2.rectangle(self.image, (self.frameR, self.frameR), (self.wCam - self.frameR, self.hCam - self.frameR),
                        (255, 0, 255), 2)

            if self.start and self.fingers:
                # stop airmouse
                if self.fingers == [0,0,0,0,0]: # all fingers closed
                    self.start = False

                # scrolling mode
                elif self.fingers == [1,1,1,1,1]: # all fingers up
                    pyautogui.scroll(25) # scroll up
                
                elif self.fingers == [0,1,1,1, 1]:
                    pyautogui.scroll(-25) # scroll down

                # only index finger is up: cursor moving Mode
                elif self.fingers == [0,1,0,0,0]:
                    # get the tip of the index finger
                    x1, y1 = self.lmList[self.index][1:]

                    # convert Coordinates
                    x3 = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
                    y3 = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))

                    # smoothen Values
                    self.clocX = self.plocX + (x3 - self.plocX) / self.smoothening
                    self.clocY = self.plocY + (y3 - self.plocY) / self.smoothening
                
                    # move Mouse
                    autopy.mouse.move(self.wScr - self.clocX, self.clocY)
                    cv2.circle(self.image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    self.plocX, self.plocY = self.clocX, self.clocY
                    
                # both index and middle fingers are up: Clicking Mode
                elif self.fingers == [0,1,1,0,0]:
                    # find distance between index and middle fingers
                    length, img, lineInfo = self.handDetector.findDistance(self.index, self.middle, self.image)

                    # click mouse if distance short
                    if length < self.click_threshold:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]),
                        15, (0, 255, 0), cv2.FILLED)
                        autopy.mouse.click()
                
                # enable gesture mode
                elif self.fingers == [1,1,1,0,0]: # if thump, index and middle finger are up
                    self.gesture_enabled = True

            else:
                # start airmouse
                if self.fingers == [1,0,0,0,0]: # if only thump is up
                    self.start = True       
    
    def fingers_detection(self):
        """
        Detects the fingers
        """
        _, self.image = self.cap.read()
        # find hands
        self.image = self.handDetector.findHands(self.image)
        # get finger coordinates
        self.lmList = self.handDetector.findLandmarkPosition(self.image)

        # call finger operation for 
        self.fingers_operation()

    def start_detection(self):
        """
        Start hand detection by defaulting to fingers mode
        """
        # while the camera is opening
        while self.cap.isOpened():
            # fingers mode
            if self.gesture_enabled is False:
                self.fingers_detection()

            # gesture mode
            elif self.gesture_enabled:
                self.gesture_detection()

            # end if key is q
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            cv2.imshow("Live Detection", self.image)

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    mc = MouseController(pre_trained=True, gesture_model_file='gb-12-33-54-01-08-2021')
    mc.start_detection()

if __name__ == "__main__":
    main()