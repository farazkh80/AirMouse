"""
A classification model build with scikit-learn, mediapipe and 
openCV for live gesture detection from webcam
"""

# import dependencies
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
import mediapipe as mp 
import cv2
from datetime import datetime
from copy import Error
import traceback


class GestureDetector(object):

    def __init__(self, dataset_file="hands-coords.csv"):
        """
        Gesture Detector Constructor
        :param dataset_file : name of the dataset file located in /data 
        """
        dataset_path = os.path.join('data', dataset_file)
        self.df = pd.read_csv(dataset_path)

        self.X = self.df.drop('class', axis=1) # features
        self.y = self.df['class'] # target value

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,
                                                                    test_size=0.3, random_state=1234)
        # model definitions
        self.models = {
            'lr': LogisticRegression(),
            'rf': RandomForestClassifier(),
            'gb': GradientBoostingClassifier()
        }
        self.detection_model_name = None # detected model name
        self.detection_model = None # detected model
        
        self.pipelines = {} # pipelines
        self.fit_models = {} # trained models
        self.model_accuracy = {} # models accuracy dict

        self.mp_drawing = mp.solutions.drawing_utils # drawing helper
        # hands solutions
        self.mp_hands = mp.solutions.hands 
        self.hand = self.mp_hands.Hands(max_num_hands=1)
        self.landmarks = None # hand landmarks

        self.image = None # image for detections
        self.detecting = False # whether model is detecting
        
        self.gesture_class = None # detected gesture class
        self.gesture_prob = None # gesture probs
        self.best_prob = None  # detected gesture prob

    def get_data_summary(self, show=False):
        """
        Data summary displayer
        :param show: if True will print the info and vice versa 
        """
        data_shape = self.df.shape
        df_count = self.df.groupby(self.df['class']).count()
        train_test_shape = {
            "X_train shape": self.X_train.shape,
            "y_train shape": self.y_train.shape,
            "X_test shape": self.X_test.shape,
            "y_test shape": self.y_test.shape
        }

        if show:
            print(f"data shape: {data_shape}")
            for key, val in train_test_shape.items():
                print(f"\t{key}: {val}")     
            print(df_count)

        return data_shape, train_test_shape, df_count
    
    def build_pipeline(self, normalize=True):
        """
        Pipeline builder if we need multiple classification model
        :param normalize: if True the data will be normalized and vice versa
        """
        print("Pipeline Building Started")

        self.pipelines = {}
        for algo, model in self.models.items():
            if normalize:
                self.pipelines[algo] = make_pipeline(StandardScaler(), model)
            else:
                self.pipelines[algo] = model

        print(f"Pipeline Building Status: SUCCESSFUL")
    
    def fit_pipeline_data(self):
        """
        Fits train data to the pipelines
        """
        print("Data Fitting Started")

        self.fit_models = {}
        for algo, pipeline in self.pipelines.items():
            model = pipeline.fit(self.X_train, self.y_train)
            self.fit_models[algo] = model

        print(f"Data Fitting Status: SUCCESSFUL")
            
    def calc_accuracy(self):
        """
        Calculates accuracy of different models
        """
        print("Accuracy Calculation Started")

        self.model_accuracy = {}
        for algo, model in self.fit_models.items():
            yhat = model.predict(self.X_test)
            self.model_accuracy[algo] = accuracy_score(self.y_test, yhat)
        
        print(f"Accuracy Calculation Status: SUCCESSFUL")
        return self.model_accuracy

    def save_model(self):
        """
        Saves the best detection model
        """
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S-%d-%m-%Y")
        file_name = str(self.detection_model_name + "-" + current_time)
        file_path = os.path.join('models', file_name)

        with open(file_path, 'wb') as f:
            pickle.dump(self.fit_models[self.detection_model_name], f)
    
    def load_model_from_file(self, from_file=False, file_name=None):
        """
        Loads the detection model from file
        :param from_file: if True loads from file, if False passes
        :param file_name: model's file name
        """
        if from_file and file_name is not None:
            file_path = os.path.join('models', file_name)
            with open(file_path, 'rb') as f:
                self.detection_model = pickle.load(f)

        print("Model Loaded")

    def get_and_set_best_model(self, save=True):
        """
        Set and return the best detection model
        :param save: if True save the model
        """
        self.detection_model_name = max(self.model_accuracy, key=self.model_accuracy.get)
        self.detection_model = self.fit_models[self.detection_model_name]
        if save:
            try:
                self.save_model()
                print("Saved the best model")
            except Error as e:
                print(e)

        return self.detection_model
  
    def train(self, multi_model= True, model_name=None, normalize=True):
        """
        Trains data by building pipelines and fitting training data to them
        :param multi_model: if True we train all models, if False train only model_name
        :param model_name: name of the model could be 'lr', 'rf', 'gb'
        :param normalize: if True data will be normalized
        """
        print("Training Started")

        if multi_model is False and model_name is not None:
            try:
                self.models =  {model_name: self.models[model_name]}
            except Error as e:
                print(e)
   
        self.build_pipeline(normalize)
        self.fit_pipeline_data()
        self.calc_accuracy()
        self.get_and_set_best_model()

        print(f"Training Status: SUCCESSFUL")

    def detect_landmarks(self, cap):
        """
        Detects hand landmarks
        :param cap: current video capture
        """
        # Recolor the image
        _, frame = cap.read()
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image.flags.writeable = False
        # Make Detections
        results = self.hand.process(self.image)

        # Landmark detection
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                
        if results.multi_hand_landmarks:
            self.landmarks = results.multi_hand_landmarks
            for handLms in self.landmarks:
                self.mp_drawing.draw_landmarks(self.image, handLms, self.mp_hands.HAND_CONNECTIONS)
        else:
            self.landmarks = None

    def make_detections_with_cv(self, cap, width=640, height=480, detection_threshold=0.90):
        """
        Make detections with the detection model on live frames
        :param cap: the camera frame
        :param width: frame width
        :param height: frame height
        :param detection_threshold: minimum probability for a valid detection
        """
        self.detect_landmarks(cap)

        # Get status box
        cv2.rectangle(self.image, (0,0), (250, 60), (245, 117, 16), -1)

        if self.landmarks:
            # Export coordinates
            try:
                # Extracting hand landmarks
                detected_hand = self.landmarks[0].landmark
                row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in detected_hand]).flatten())

                # Make Detections
                X = pd.DataFrame([row])
                self.gesture_class = self.detection_model.predict(X)[0]
                self.gesture_prob = self.detection_model.predict_proba(X)[0]
                self.best_prob = round(self.gesture_prob[np.argmax(self.gesture_prob)],2)

                if self.gesture_class and self.best_prob > detection_threshold:
                    self.detecting = True
                    # Display Class
                    cv2.putText(self.image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(self.image, self.gesture_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                    cv2.putText(self.image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(self.image, str(self.best_prob)
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                else:
                    self.detecting = False

                    # Display Message
                    cv2.putText(self.image, "No Detection"
                                , (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            except:
                traceback.print_exc()  

        else:
            self.detecting = False

            # Display Message
            cv2.putText(self.image, "No Detection"
                        , (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        # Display Message
        cv2.putText(self.image, "Press q for closing"
                    , (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
        return self.image

    def get_detected_gesture(self):
        """
        Returns gesture class and best prob
        """
        if self.detecting:
            return self.gesture_class, self.best_prob
        else:
            return None, None

def main():
    gd = GestureDetector()
    gd.train()
    gd.get_and_set_best_model()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while cap.isOpened():
        gd.make_detections_with_cv(cap)
        gesture, gesture_prob = gd.get_detected_gesture()
            
        print(f"gesture: {gesture}\t prob: {gesture_prob}")

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()