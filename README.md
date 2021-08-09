# AirMouse

A hands-free PC control module using live hand gesture detection and finger movement tracking.
With AirMouse, basic mouse operations such as clicking, moving cursor, scrolling and zooming are possible without even touching the mouse. Moreover, AirMouse' gesture detection extension adds on the possibility of user defined gestures and computer operations such as playing your favorite Spotify playlist by showing a thumps up!

## Motivation

AirMouse is a personal project inspired by Google's Mediapipe Machine Learning solutions for hand detection. Using the coordination data collected by Mediapipe solutions, AirMouse takes the human-computer interaction experience to the next level by making it more personalized and creative.

## Installation

Use the package manager pip to install the dependencies.

```bash
pip install -r requirements.txt
```

Then you can simply run

```bash
python MouseController.py
```

## How it Works

AirMouse file structure:

```
AirMouse
│   README.md
│   MouseController.py
|   HandDetectorModule.py
|   GestureDetectorScikit.py
|   GestureDetectorTorch.py
│
└───data
│   │   hands-coords.csv
│
│
└───models
|   |   gb-14-14-39-01-08-2021
|   |   torch-mc-12-18-38-08-08-2021
|
|
│
└───prototypes
    |   GestureClassifier.ipynb
    |   GestureClassifierNet.ipynb
    |   HandDataCollector.ipynb
    |   HolisticDataCollector.ipynb
```

There are two modes that AirMouse operates with:

- Basic Mode

  - cursor movement
  - mouse click
  - scroll up and down

- Gesture Mode
  - zooming in and out
  - custom automations

`MouseController.py` is the main controller that calls upon `HandDetectorModule.py` for hand detection and finger position tracking when in Basic Mode and calls any of `GestureClassifierTorch.py` and `GestureClassifierScikit.py` modules for detecting the gesture being shown when in Gesture Mode.

When in Basic Mode:
`HandDetectorModule.py` is responsible for detecting the position of the hand using OpenCV and then extract the hand landmark coordinates using Mediapipe Machine Learning Solutions `HandDetectorModule.py` can then determine the finger tip position and whether the fingers are up or down for automating basic operations such as moving mouse cursor around with the movement of index finger or clicking when index finger and middle finger are close to each other.

When in Gesture Mode:
any of `GestureClassifierTorch.py` and `GestureClassifierScikit.py` can use `/data/hands-coords.csv` data for training a Multi Classification Neural Network Model(Pytorch) or a pre-defined Classification model such as Logistic Regression or Random Forest Classifier(Scikit-learn) and predicting the gesture being capture by OpenCV webcam. Upon predicting the gesture class, the `MouseController.py` can perform personalized computer operations such as playing a specific video or showing the current weather outside.

## Usage

![Demo.mp4](demo.gif)

##### Basic Mode Operations:

#### Gesture Mode Operations:

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

MIT
