[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"

## Dog Breed Classifier

This repository contains my solution for Udacity's second Deep Learning Nanodegree project. After achieving all the objectives of the "Dog Breed Classifier" project, an algorithm results that, given an image, detects whether there is a human face or a dog in it. In case it is a dog, the algorithm classifies the breed of the dog.

### Main objectives

The main objectives of the project are:
1. Given a [Cascade Classifier](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html), be able to detect faces in images.
2. Use the VGG-16 model to dectect dogs in images, and give percentages of detected dogs in the downloaded datasets.
```
Percentage of human files with detected dogs: 0 %
Percentage of dog files with detected dogs: 84 %
```
3. Create a Convolutional Neural Network to Classify Dog Breeds
* Specify the data loaders
* Define the model architecture suitable for the classifying problem
* Train and Validate the model
* Test the model in order to have more than a 10% of accuracy
```
Test Loss: 3.805343
Test Accuracy: 13% (111/836)
```
4. Use _Transfer Learning_ to classify Dog Breeds
* Choose the Convolutional Neural Network architecture that best suits the classifying problem
* Modify the network in order to classify dog breeds
* Train and Validate the model
* Test the model in order to have more than a 60% of accuracy
```
Test Loss: 2.204232
Test Accuracy: 64% (538/836)
```

### Get Started

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/deep-learning-v2-pytorch.git
		cd deep-learning-v2-pytorch/project-dog-classification
	```
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

Notice that `dogImages/` and `lfw/` folders are __NOT__ included in the repository

`Python` must be installed.
The following packages must be installed:

```
opencv-python
jupyter
matplotlib
pandas
numpy
pillow
scipy
tqdm
seaborn
scikit-learn
scikit-image
h5py
ipykernel
```
