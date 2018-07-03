# **Traffic Sign Recognition**

## Writeup

## Brett Fotheringham

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/testset.png "Training Set"
[image2]: ./examples/validationset.png "Validation Set"
[image3]: ./examples/testsetreal.png "Testing Set"

[image4]: ./DoNotEnter.jpg "Traffic Sign 1"
[image5]: ./RoadWork.jpg "Traffic Sign 2"
[image7]: ./50km_hr_.jpg "Traffic Sign 3"
[image6]: ./CautionSign.jpg "Traffic Sign 4"
[image8]: ./GoStraightOrRight.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The first image is a bar chart showing the distribution of the data across the 43 different classes for the training set. The validation set, as seen in image 2, has the exact same distribution as the training set. The test set however has a unique distribution as seen in image 3.

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The only major step of preprocessing done to the images was to normalize them. Initially, I used no preprocessing on the images and was achieving validation accuracies of only 50-60%. After normalizing the images, that accuracy improved significantly. In fact, I mistakenly forgot to normalize the test set and once I changed, that I achieved significantly higher accuracies on the test set.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 9x9     	| 1x1 stride, same padding, outputs 24x24x6 	|
| RELU					|												|
| Convolution 7x7     	| 1x1 stride, same padding, outputs 18x18x12 	|
| RELU					|												|
| Convolution 9x9     	| 1x1 stride, same padding, outputs 10x10x18 	|
| RELU					|												|
| Max Pooling 2x2 | outputs 5x5x18
| Fully connected		| 3 layers as following (input->450->250->120->84->43->output) |
| Softmax				| cross entropy with logits       									|
|	Mean Reduction  		|	loss operation on cross entropy											|
|	Optimizer					|	AdamOptimizer											|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

As a beginning to the discussion of model training, the following parameters used for the model were: a batch size of 64, the Adam Optimizer, 20 epochs (or less). No dropout technique was used, however, this may have helped improved the identification of my five additional images. The model was trained as I said for 20 epochs or less. What I did, was to break out of the training forloop once the validation accuracy was above a certain point. While this may not make sense in all applications, it does when trying to achieve higher validation accuracies as will be discussed in the next step.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.954
* test set accuracy of 0.922

The original architecture used was that provided by the LeNet lab, the LeNet-5 architecture. It was chosen because it was the easiest starting point as I was given a tutorial for how to adapt it to the 32x32 traffic sign images. This made it a fantastic tool to experiment with. Nevertheless, the LeNet-5 architecture struggles with more detailed images liked traffic signs (as opposed to letters) because of it's use of max-pooling. Because of the three max-pooling layers a large amount of information is lost which was reducing the model's accuracy.

Prior to coming to this conclusion, an additonal hidden layer was added to the network to experiment on the potential effects that it would have on the results. After implementing the additional layer, and reading about the hidden layers of neural network, I came to the conclusion that this layer was unnecessary and made little to no changes in the results. After thinking about the current LeNet-5 architecture however, I discovered, as mentioned above, that max pooling could be replaced with larger or more convolutions. Once I removed the max pooling and implemented the larger convolutions, the model speed dropped dramatically. Nevertheless, there was a considerable rise in the accuracy on both the validation and test sets.

With regards to the hyperparameters, the learning rate was halved to 0.0005, the epochs were doubled to 20, and the batch size was halved. However, if the validation accuracy jumped above 94%, the training forloop was broken. These parameters  were adjusted by trial and error. The learning rate was specifically reduced because after reading, I realized I might miss the minima for the error that results from gradient descent if my learning rate is too large.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

These images should be incredibly easy to classify. Once they are cropped, and reshaped to 32x32, they become the 
quintessential examples of their class. The only explanation for the programs inability to classify them is incorrect data.
That is an exaggeration, but nevertheless it is the strongest conclusion I can come to.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road work      		| 20 km/h									|
| Go Straight or Right     			| 50 km/h										|
| General Caution					| 20 km/h										|
| 50 km/h	      		| Keep Right					 				|
| Do Not Enter			| 20 km/h      							|


The model was able to correctly guess 0 of the 5 traffic signs, which gives an accuracy of 0%. This compares terribly with the test set accuracy of over 90%. The only explanation I can come up with is that my images have been processed incorrectly. Nevertheless, after digging into pickled data and plotting some of the images as well as my own, the two appear to be formatted in the same way. There is a major piece of the puzzle missing here with my images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

As you will see below the model wasn't very sure about any of its classifications. The major of the probabilites are close to each other indicating a low level of confidence in the classifications.

Image One:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .25         			| 20 km/r 									|
| .22     				| Go Straight or left 										|
| .19					| Ahead Only											|
| .17	      			| Keep right					 				|
| .17				    | Turn right ahead      							|





