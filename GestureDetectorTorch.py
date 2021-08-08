"""
A classification model build with PyTorch, mediapipe and 
openCV for live gesture detection from webcam
"""

# import dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mediapipe as mp 
import cv2
import time
import os
import traceback
from datetime import datetime
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class GestureDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        return len(self.y_data)
    
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

class GestureClassifierNet(nn.Module):
    def __init__(self, num_feature, num_class):
        super(GestureClassifierNet, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x

class GestureDetector(object):

    def __init__(self, dataset_file="hands-coords.csv"):
        """
        Gesture Detector Constructor
        :param dataset_file : name of the dataset file located in /data 
        """
        dataset_path = os.path.join('data', dataset_file)
        self.df = pd.read_csv(dataset_path)

        # Encode the class
        self.le = LabelEncoder()
        self.le.fit(self.df['class'])
        self.df['class_encoded'] = self.le.transform(self.df['class'])

        self.X = self.df.drop(['class', 'class_encoded'], axis=1) # features
        self.y = self.df['class_encoded'] # target value

        # Data Preprocessing and Scaling fit
        self.dataset_setup()

        '''
        Model Definitions
        '''
        self.EPOCHS = 100
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 0.0007
        self.NUM_FEATURES = len(self.X.columns)
        self.NUM_CLASSES = max(self.y) + 1

        self.accuracy_stats = {
            'train': [],
            "val": []
        }
        self.loss_stats = {
            'train': [],
            "val": []
        }

        self.test_accuracy = 0
        self.test_loss = 0

        self.detection_model_name = 'torch-mc' # detected model name
        self.detection_model = None # detection model

        '''
        Other definitions
        '''
        self.mp_drawing = mp.solutions.drawing_utils # drawing helper
        self.mp_hands = mp.solutions.hands # hands solutions
        self.hand = self.mp_hands.Hands(max_num_hands=1) # hands construction
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
    
    def dataset_setup(self):
        """
        Dataset Preprocessing and Transform
        """
        # Divide it to trainval and test splits
        self.X_trainval, self.X_test, \
        self.y_trainval, self.y_test = train_test_split(
                                                        self.X,
                                                        self.y, 
                                                        stratify=self.y,
                                                        test_size=0.3, 
                                                        random_state=69
                                                        )


        # Split train into train-val
        self.X_train, self.X_val, \
        self.y_train, self.y_val = train_test_split(
                                            self.X_trainval,
                                            self.y_trainval,
                                            test_size=0.1,
                                            stratify=self.y_trainval, 
                                            random_state=21
                                            )

        # Scale the data
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
        self.X_val, self.y_val = np.array(self.X_val), np.array(self.y_val)
        self.X_test, self.y_test = np.array(self.X_test), np.array(self.y_test)

        self.train_dataset = GestureDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).long())
        self.val_dataset = GestureDataset(torch.from_numpy(self.X_val).float(), torch.from_numpy(self.y_val).long())
        self.test_dataset = GestureDataset(torch.from_numpy(self.X_test).float(), torch.from_numpy(self.y_test).long())

    def set_weighted_sampling(self):
        """
        Weighted sampling the training datasets
        """
        def get_class_distribution(obj, max_num_class):
            count_dict = {}
            for i in range(max_num_class+1):
                count_dict[i] = 0
            
            for i in obj:
                count_dict[i] += 1
                    
            return count_dict

        target_list = []
        for _, t in self.train_dataset:
            target_list.append(t)
            
        target_list = torch.tensor(target_list)
        target_list = target_list[torch.randperm(len(target_list))]

        class_count = [i for i in get_class_distribution(self.y_train, int(max(target_list))).values()]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

        self.class_weights_all = class_weights[target_list]
        self.weighted_sampler = WeightedRandomSampler(
            weights=self.class_weights_all,
            num_samples=len(self.class_weights_all),
            replacement=True
        )

    def model_setup(self):
        """
        Model and Data Loader Setup
        """
        print("MODEL SETUP STARTED")
        self.detection_model = GestureClassifierNet(num_feature = self.NUM_FEATURES, num_class = self.NUM_CLASSES)

        self.set_weighted_sampling()
        self.train_loader = DataLoader(
                                dataset=self.train_dataset,
                                batch_size=self.BATCH_SIZE,
                                sampler=self.weighted_sampler,
                                drop_last=True
                                )
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=1)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=1)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.detection_model.parameters(), lr=self.LEARNING_RATE)
        print("MODEL SETUP FINISHED")

    def multi_acc(self, y_pred, y_test):
        """
        Accuracy Function
        """
        y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
        
        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)
        
        acc = torch.round(acc * 100)
        
        return acc
    
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

    def save_model(self):
        """
        Saves the best detection model
        """
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S-%d-%m-%Y")
        file_name = str(self.detection_model_name + "-" + current_time)
        file_path = os.path.join('models', file_name)

        with open(file_path, 'wb') as f:
            pickle.dump(self.detection_model, f)
  
    def train(self):
        """
        Sets up the model and trains it over the training and validation dataset
        """
        print("Training Started")

        # Model Setup
        self.model_setup()

        # Training
        for e in range(self.EPOCHS):

            train_epoch_loss = 0
            train_epoch_acc = 0

            # TRAINING
            self.detection_model.train()
            for X_train_batch, y_train_batch in self.train_loader:
                self.optimizer.zero_grad()
                
                y_train_pred = self.detection_model(X_train_batch)
                
                train_loss = self.criterion(y_train_pred, y_train_batch)
                train_acc = self.multi_acc(y_train_pred, y_train_batch)
                
                train_loss.backward()
                self.optimizer.step()
                
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            # VALIDATION    
            with torch.no_grad():
                
                val_epoch_loss = 0
                val_epoch_acc = 0
                
                self.detection_model.eval()
                for X_val_batch, y_val_batch in self.val_loader:
                    y_val_pred = self.detection_model(X_val_batch)
                                
                    val_loss = self.criterion(y_val_pred, y_val_batch)
                    val_acc = self.multi_acc(y_val_pred, y_val_batch)
                    
                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            self.loss_stats['train'].append(train_epoch_loss/len(self.train_loader))
            self.loss_stats['val'].append(val_epoch_loss/len(self.val_loader))
            self.accuracy_stats['train'].append(train_epoch_acc/len(self.train_loader))
            self.accuracy_stats['val'].append(val_epoch_acc/len(self.val_loader))
            if e%10==0:
                print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(self.train_loader):.5f} | Val Loss: {val_epoch_loss/len(self.val_loader):.5f} | Train Acc: {train_epoch_acc/len(self.train_loader):.3f}| Val Acc: {val_epoch_acc/len(self.val_loader):.3f}')
        print(f"Training Status: SUCCESSFUL")

    def evaluate(self):
        """
        Evaluates Model
        """
        y_pred_list = []
        with torch.no_grad():
            test_loss=0
            test_acc=0
            self.detection_model.eval()
            for X_batch, Y_batch in self.test_loader:
                y_test_pred = self.detection_model(X_batch)
                _, y_pred_tags = torch.max(y_test_pred, dim = 1)
                y_pred_list.append(y_pred_tags.cpu().numpy())

                test_it_loss = self.criterion(y_test_pred, Y_batch)
                test_it_acc = self.multi_acc(y_test_pred, Y_batch)
                
                test_loss += test_it_loss.item()
                test_acc += test_it_acc.item()

            self.test_loss = (test_loss/len(self.test_loader))
            self.test_accuracy = (test_acc/len(self.test_loader))

        # Create dataframes
        train_val_acc_df = pd.DataFrame.from_dict(self.accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        train_val_loss_df = pd.DataFrame.from_dict(self.loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        # Plot the dataframes
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
        sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
        sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

        print(f"\n\nTest Loss:{self.test_loss} \t Test Acc:{self.test_accuracy}")

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
                X = self.scaler.transform(X)
                X = np.array(X)

                X = torch.tensor(X, dtype=torch.float32)

                with torch.no_grad():
                    self.detection_model.eval()
                    y_test_pred = self.detection_model(X)
                    _, y_pred_tags = torch.max(y_test_pred, dim = 1)
                    self.gesture_class = self.le.inverse_transform(y_pred_tags.cpu().numpy())[0]

                # Display Class
                cv2.putText(self.image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(self.image, self.gesture_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                self.detecting=True
            
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
    gd.evaluate()

    gd.save_model()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while cap.isOpened():
        image = gd.make_detections_with_cv(cap)
        cv2.imshow("Live Detection", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()