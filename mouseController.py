from HandDetectorModule import HandDetector
import cv2
import webbrowser
import numpy as np
import GestureDetectorScikit as gds
from GestureDetectorTorch import GestureClassifierNet, GestureDataset
import GestureDetectorTorch as gdt
import autopy
import pyautogui
import pywhatkit
import time

class MouseController():
    def __init__(self, gesture_dataset_file='hands-coords.csv', pre_trained=False, gesture_model_file=None, library='PYTORCH', width=640, height=480):
        """
        Constructor
        :param gesture_dataset_file: gesture dataset file name
        :param pre_trained: whether the model will be uploaded from a file
        :param gesture_model_file: gesture model file name
        :param library: whether to train using "SCIKIT" or "PYTORCH"
        :param width: camera frame width
        :param height: camera frame height
        """

        self.gesture_dataset_file = gesture_dataset_file # gesture dataset file
        self.pre_trained = pre_trained # pretrained model flag from a file
        self.gesture_model_file = gesture_model_file # pretrained model file name
        self.library = library # model library

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

        self.handDetector =  HandDetector(maxHands=1)
        self.lmList = [] # list of landmark coordinates
        self.fingers = None # binary list of fingers indicating they are up or down
        # Finger tip ids
        self.thump = 4
        self.index = 8
        self.middle = 12
        self.ring = 16
        self.pinky = 20

        self.gesture = None # classified gesture
        self.operation_mode = 'READY'
        self.gesture_prob = None # classified gesture probability
        
    def setup(self):
        # set model library
        if self.library == "SCIKIT":
            self.gesture_detector = gds.GestureDetector(self.gesture_dataset_file)

        else:
            self.gesture_detector = gdt.GestureDetector(self.gesture_dataset_file)

        # if the model is to be loaded from a file and is not already trained
        if self.pre_trained and self.gesture_model_file is not None:
            self.gesture_detector.load_model_from_file(True, self.gesture_model_file)
        # if the model is to be trained and is not already trained
        else:
            self.gesture_detector.train()

    def gesture_operation(self):
        """
        Automated operation handler based on the detected self.gesture
        """

        # Operation: zoom in
        if self.gesture == 'OPEN':
            self.operation_mode = "Zoom In"
            pyautogui.keyDown('ctrl')
            pyautogui.keyDown('alt')
            pyautogui.scroll(25)
            pyautogui.keyUp('alt')
            pyautogui.keyUp('ctrl')

        # Operation: zoom out
        elif self.gesture == 'CLOSE':
            self.operation_mode = "Zoom Out"
            pyautogui.keyDown('ctrl')
            pyautogui.keyDown('alt')
            pyautogui.scroll(-25)
            pyautogui.keyUp('alt')
            pyautogui.keyUp('ctrl')

        # Operation: Play country songs on youtube
        elif self.gesture == 'Thumps-Up':
            self.operation_mode = "Play Happy Songs"
            time.sleep(1)
            pywhatkit.playonyt("https://www.youtube.com/watch?v=QTpxT-Wie1s&t=919s")
            self.gesture_enabled = False

        # Operation: Play Sad Songs
        elif self.gesture == 'Thumps-Down':
            self.operation_mode = "Play Happy Songs"
            time.sleep(1)
            pywhatkit.playonyt("https://www.youtube.com/watch?v=Y_oD111dK7c")
            self.gesture_enabled = False

        # Operation: Open Swag Pic
        elif self.gesture == 'Swag':
            self.operation_mode = "Swag Pic"
            webbrowser.open("https://wallpaper.dog/large/17215240.jpg")
            self.gesture_enabled = False

        # Operation: Exit Gesture Mode
        elif self.gesture == 'Peace':
            self.operation_mode = "Exit Gesture Mode"
            time.sleep(1)
            self.gesture_enabled = False

        else:
            self.operation_mode = None
            

    def gesture_detection(self):
        """
        Detects the gesture class
        """
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
                    self.operation_mode='Stop AirMouse'
                    self.start = False

                # scrolling mode
                elif self.fingers == [1,1,1,1,1]: # all fingers up
                    self.operation_mode='Scroll Up'
                    pyautogui.scroll(40) # scroll up
                
                elif self.fingers == [0,1,1,1, 1]:
                    self.operation_mode='Scroll Down'
                    pyautogui.scroll(-40) # scroll down

                # cursor moving Mode
                elif self.fingers == [0,1,0,0,0]: # only index finger is up: 
                    self.operation_mode='Cursor Moving'
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
                    
                # Clicking Mode
                elif self.fingers == [0,1,1,0,0]: # both index and middle fingers are up and touching 
                    self.operation_mode='CLICK'
                    # find distance between index and middle fingers
                    length, img, lineInfo = self.handDetector.findDistance(self.index, self.middle, self.image)

                    # click mouse if distance short
                    if length < self.click_threshold:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]),
                        15, (0, 255, 0), cv2.FILLED)
                        autopy.mouse.click()
                
                # enable gesture mode
                elif self.fingers == [0,0,0,0,1]: # if only pinky finger is up
                    self.operation_mode='Enter Gesture Mode'
                    self.gesture_enabled = True

            else:
                # start airmouse
                if self.fingers == [1,0,0,0,0]: # if only thump is up
                    self.operation_mode='Start AirMouse'
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
        self.setup()
        # while the camera is opening
        pTime = 0
        while self.cap.isOpened():
            _, self.image = self.cap.read()
            self.image.flags.writeable = True
        

            # fingers mode
            if self.gesture_enabled is False:
                self.fingers_detection()
                # Put status box
                cv2.rectangle(self.image, (0,0), (350, 60), (245, 117, 16), -1)

                # Display Class
                cv2.putText(self.image, 'Basic'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(self.image, self.operation_mode
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # gesture mode
            elif self.gesture_enabled:
                self.gesture_detection()
                # Put status box
                cv2.rectangle(self.image, (0,0), (350, 60), (245, 117, 16), -1)
                # Display Class
                cv2.putText(self.image, 'Gesture'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(self.image, f"{self.gesture} ({self.operation_mode})"
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Display Frame Rate
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(self.image, 'FPS:'
                        , (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(self.image, str(int(fps))
                        , (60,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)        

            # end if key is q
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            cv2.imshow("Live Detection", self.image)

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    mc = MouseController(pre_trained=True, gesture_model_file='torch-mc-16-52-56-08-08-2021', library="PYTORCH")
    mc.start_detection()

if __name__ == "__main__":
    main()