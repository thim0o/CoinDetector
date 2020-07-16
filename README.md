# A VGG16-based coins detector

Authors: [Khanh Vu](https://github.com/khanhvu207), [Thimo Blom](https://github.com/thim0o), Hannah Stone.  
Institute: Vrije Universiteit Amsterdam.

## Description
### 1. Images acquisition (tools: OpenCV, Python)
1. Capture raw images from the webcam.
2. Convert RGB to HSV color space.  
![HSV](https://github.com/thim0o/CoinDetector/blob/master/img/hsv.JPG)  
3. Apply [median blur](https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html) in order to cancel noises.
4. Apply [image thresholding](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html) & [background subtraction](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html) to capture existing coins.  
![params](https://github.com/thim0o/CoinDetector/blob/master/img/params.JPG)![thresholding](https://github.com/thim0o/CoinDetector/blob/master/img/img_thresholding.JPG)   
5. Make bounding boxes and crop out coins.  
![extracted_coins](https://github.com/thim0o/CoinDetector/blob/master/img/extracted_coins.JPG)  

### 2. Convolutional neural network - The VGG16 (Pre-trained model)
***From https://neurohive.io/en/popular-networks/vgg16/***
> VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous model submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another.  
![VGG16](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

We made few adjustions to the existing VGG16 model from Keras:  
1. We kept only the convolutional layers so we removed the last 4 layers (flatten + 2 x 4096 fully-connected + 1000 classes prediction layers).   
2. Added some fully-connected layers together with a batch_normalization layer and a dropout layer.
![model](https://github.com/thim0o/CoinDetector/blob/master/model_plot.png)

### 3. Model training (tools: Google Colab)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15Yoca54IK-rvgcD8L71iymNVYCUOtcKG?usp=sharing)
* Dataset:
  * Total images: **2923 (~500 images per class)**.
  * 6 classes: '10cent', '1euro', '20cent', '2euro', '50cent', '5cent'.
  * Train/Validation ratio: 7:3.
  * Data augmentation: Rotation and Flip.

* Model:
  * Optimizer: Adam - Learning rate = 0.0001.
  * Loss function: 'categorical_crossentropy'.
  * **Transfer learning:** We froze all the layers except for the last 7 hidden layers to accelerate the training process.
  * Metrics tracker: Tensorboard.

### 4. Deploy the model and classify coins (locally, tools: Jupyter notebook)
We downloaded the model params (model.h5) and ran it in our machine - [source code](https://github.com/thim0o/CoinDetector/blob/master/demo.ipynb).  
![demo](https://github.com/thim0o/CoinDetector/blob/master/img/demo.gif)  

## Evaluation
See our [project report](https://github.com/thim0o/CoinDetector/blob/master/Real-time-CNN-coin-classification-with-OpenCV-data-acquisition.pdf) for more details.
