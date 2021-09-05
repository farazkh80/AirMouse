# AirMouse

A hands-free PC control module using live hand gesture detection and finger movement tracking.
With AirMouse, basic mouse operations such as clicking, moving cursor, scrolling and zooming are possible without even touching the mouse. Moreover, AirMouse' gesture detection extension adds on the possibility of user defined gestures and computer operations such as playing your favorite Spotify playlist by showing a thumps up!

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

AirMouse is a personal project inspired by Google's Mediapipe Machine Learning solutions for hand detection. Using the coordination data collected by Mediapipe solutions, AirMouse takes the human-computer interaction experience to the next level by making computer control more personalized and creative.

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
`HandDetectorModule.py` is responsible for detecting the position of the hand using OpenCV and then extract the hand landmark coordinates using Mediapipe Machine Learning Solutions `HandDetectorModule.py` can then determine the finger tip position and whether the fingers are up or down for automating basic operations such as moving mouse cursor around with the movement of index finger or clicking when index finger and middle finger are close to each other.

When in **Gesture** Mode:
any of `GestureClassifierTorch.py` and `GestureClassifierScikit.py` can use `/data/hands-coords.csv` data for training a Multi Classification Neural Network Model(Pytorch) or a pre-defined Classification model such as Logistic Regression or Random Forest Classifier(Scikit-learn) and predicting the gesture being captured by OpenCV webcam. Upon predicting the gesture class, the `MouseController.py` can perform personalized computer operations such as playing a specific video or showing the current weather outside.

# Current Issues

- Searching dataset initialization takes place before each new search, reducing the searching speed
- Model loading from checkpoint also happens before each summarization task
- Constant, non-configurable `text_max_token_len` of 512 and `summary_max_token_len` of 128 as suggested by the original [T5](https://arxiv.org/abs/1910.10683) paper, limit the word-range of input and output text

# Contributing

Contributions are always welcome!

A detailed guide about project specifics and in-depth architecture analysis will be released soon.

# Roadmap

- Fixing Issues:

  - Avoid searching dataset initialization from happening before each search by either running both the text summarization and searching engines in a parallel setup or storing search dataset in a database.
  - Load model from checkpoint only once at the beginning of the program or just before the first summarization prompt
  - Add configurable `text_max_token_len` and `summary_max_token_len` to increase text word-range flexibility.

- New Features:
  - Fine-tune T5 Models with higher number of parameters such as `t5-large` and `t5-3B` for better results.
  - Add other NLP models such as `BERT` and `ALBERT` for performance comparison.
  - Add a web news scrapper to maintain an up-to-date version of latest most popular `n` news.

# Acknowledgements

T5 Text Summarizer

- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hugging Face T5 API](https://huggingface.co/transformers/model_doc/t5.html)
- [PyTorch Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html)
- [Text Summarization with T5](https://www.youtube.com/watch?v=KMyZUIraHio)

Search Engine

- [Building a Search Engine with C++](https://www.education-ecosystem.com/nikos_tsiougkranas/ljJg5-how-to-build-a-search-engine-in-c/yDd46-intro-how-to-build-a-search-engine-in-c/)

# License

This project is licensed under the terms of the [MIT](https://en.wikipedia.org/wiki/MIT_License) license.
