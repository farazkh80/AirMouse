# AirMouse

A hands-free PC control module using live hand gesture detection and finger movement tracking.
<br>
With AirMouse, basic mouse operations such as clicking, moving cursor and scrolling are possible without even touching the mouse. Moreover, AirMouse' gesture detection extension adds on the possibility of user defined gestures and computer operations such as playing your favorite Spotify playlist by showing a thumps up!

# Table of Contents

- [AirMouse](#airmouse)
- [Table of Contents](#table-of-contents)
- [Features](#features)
  - [1. Basic Mode Features](#1-basic-mode-features)
  - [2. Gesture Mode Features](#2-gesture-mode-features)
- [Installation/Usage](#installationusage)
- [Motivation](#motivation)
- [Project Overview](#project-overview)
- [Current Issues](#current-issues)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [License](#license)

# Features

AirMouse features consist of two main modes, **Basic** mode used for basic mouse functionality such as moving the cursor and clicking, and **Gesture** mode used for custom gesture-based operation such as playing a music playlist or browsing a specific webpage.

## 1. Basic Mode Features

- 1.1 Starting AirMouse ✅
- 1.2 Moving the Cursor 👆
- 1.3 Clicking click 🖱
- 1.4 Scrolling Up and Down ⬆⬇
- 1.5 Enabling Gesture Mode 🚀
- 1.6 Stopping AirMouse ❌

## 2. Gesture Mode Features

- 2.1 Zoom In 🔍
- 2.2 Zoom Out 🔎
- 2.3 Play Happy Songs 👍
- 2.4 Play Sad Songs 👎
- 2.5 Open Swag Picture on Browser 🤙
- 2.6 Exit Gesture Mode ❌

# Installation/Usage

1. Clone the repository and change the directory to the path the cloned repository is located at

```bash
git clone https://github.com/farazkh80/AirMouse.git
cd AirMouse
```

2. Create a virtual environment and activate it

```bash
virtualenv venv
venv/Scripts/activate
```

3. Install the project requirements

```bash
pip install -r requirements.txt
```

4. To start the AirMouse webcam simply run

```bash
python MouseController.py
```

**Note: For proper starting and usage of AirMouse refer to [Features](#features).**

# Motivation

AirMouse is a personal project inspired by Google's [Mediapipe](https://google.github.io/mediapipe/solutions/hands) Machine Learning solutions for hand detection. Using the coordination data collected by [Mediapipe](https://google.github.io/mediapipe/solutions/hands) solutions, AirMouse takes the human-computer interaction experience to the next level by making computer control more personalized and creative.

# Project Overview

AirMouse file structure:

```
AirMouse
│   README.md
│   MouseController.py #main file with definitions for each mode
|   HandDetectorModule.py #hand detector and finger distance calculator
|   GestureDetectorScikit.py #scikit learn gesture classification model
|   GestureDetectorTorch.py #Pytorch gesture classification DNN
│
└───data
│   │   hands-coords.csv #collected hand landmark coordinates
│
│
└───models #pre-trained model checkpoints
|   |   gb-14-14-39-01-08-2021
|   |   torch-mc-12-18-38-08-08-2021
|
|
│
└───prototypes
    |   GestureClassifier.ipynb #scikit learn classifier prototype for AirMouse
    |   GestureClassifierNet.ipynb #pytorch DNN prototype for AirMouse
    |   HandDataCollector.ipynb #hand landmark coordinate collector
    |   HolisticDataCollector.ipynb #holistic landmark coordinate collector
```

`MouseController.py` is the main controller that calls upon `HandDetectorModule.py` for hand detection and finger position tracking when in Basic Mode and calls either of `GestureClassifierTorch.py` and `GestureClassifierScikit.py` modules for detecting the gesture being shown when in Gesture Mode.

When in **Basic** Mode:
`HandDetectorModule.py` is responsible for detecting the position of the hand using OpenCV and then extract the hand landmark coordinates using [Mediapipe](https://google.github.io/mediapipe/solutions/hands) Machine Learning Solutions `HandDetectorModule.py` can then determine the finger tip position and whether the fingers are up or down for automating basic operations such as moving mouse cursor around with the movement of index finger or clicking when index finger and middle finger are close to each other.

When in **Gesture** Mode:
Either of `GestureClassifierTorch.py` and `GestureClassifierScikit.py` can use `/data/hands-coords.csv` data for training a Multi Classification Neural Network Model(Pytorch) or a pre-defined Classification model such as Logistic Regression or Random Forest Classifier(Scikit-learn) and predicting the gesture being captured by OpenCV webcam. Upon predicting the gesture class, the `MouseController.py` can perform personalized computer operations such as playing a specific video or showing the current weather outside.

# Current Issues

- Certain computer operations such as zooming-in and zooming-out reduces the video capture speed to 2-3 FPS, due to their complex automation.
- Gesture mode operations cannot be used consecutively without switching to basic mode in between, due to gesture classification per FPS.

# Contributing

Contributions are always welcome!

A detailed guide about project specifics and in-depth architecture analysis will be released soon.

# Roadmap

- Fixing Issues:

  - Stay in Gesture Mode unless prompted to change by user
  - Move zooming to basic mode

- New Features:
  - Add configurable gesture detection API for custom operations
  - Add holistic data collector for emotion detection
  - Add AirKeyboard

# Acknowledgements

- [Mediapipe Hand Detector](https://google.github.io/mediapipe/solutions/hands)
- [Multi Classification with PyTorch](https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab)
- [AI Body Language Decoder](https://www.youtube.com/watch?v=pG4sUNDOZFg)
- [AI Face Body and Hand Posen Estimation](https://www.youtube.com/watch?v=pG4sUNDOZFg)
- [FingerDetection](https://www.youtube.com/watch?v=NZde8Xt78Iw)

# License

This project is licensed under the terms of the [MIT](https://en.wikipedia.org/wiki/MIT_License) license.
