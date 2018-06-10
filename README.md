# **Traffic Sign Recognition** 

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

[image1]: ./examples/stat.png "Statistics"
[image2]: ./examples/image_before.png "Image before"
[image3]: ./examples/image_after.png "Image after"
[image4]: ./examples/new_images.png  "New"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points

### Data Set Summary & Exploration

#### 1. A basic summary of the data set.

The code for this step is contained in the 3d code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Exploratory visualization of the dataset.

The code for this step is contained in the 5th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed. We see that the distribution is uneven.

![alt text][image1]

Trafic signs summary statistics:
count      43.0
mean      809.0
std       627.0
min       180.0
50%       540.0
max      2010.0

### Design and Test a Model Architecture

#### 1. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 6-9 code cells of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the image color is not important.
Also, I decided to implement normalisation, dividing the image by 255.

Here is an example of a traffic sign image before grayscaling with normalisation.
![alt text][image2]

Here is an examples of a traffic sign image after grayscaling with normalisation. I tried different normalisation (diving by 255 and dividing by 128 after subtraction 128).
![alt text][image3]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the 3d code cell of the IPython notebook.  

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

#### 3. Final model architecture.

The code for my final model is located in the 11th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 RGB image                             | 
| Convolution 3x3       | 1x1 stride, valid padding, outputs 30x30x10   |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 15x15x10                 |
| Convolution 4x4       | 1x1 stride, valid padding, outputs 12x12x30   |
| RELU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 6x6x30                   |
| Dropout               | Keep prob 50%                                 |
| Flatten               | Output 1080                                   |
| Fully connected       | Output 270                                    |
| RELU	                |                                               |
| Fully connected       | Output 129                                    |
| RELU	                |                                               |
| Fully connected       | Output 43                                     |
| Softmax               |                                               |


#### 4. Training model. 

The code for training the model is located in the 12-15 cells of the ipython notebook. 

To train the model, I used an AdamOptimizer

#### 5.The approach taken for finding a solution.
The code for calculating the accuracy of the model is located in the 15-16 cells of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.946 
* test set accuracy of 0.942

Steps to find final approach:
* The first architecture that I was tried was LeNet Architecture. Sign recognition is similar to MNIST case, so i decided to try it. 
* LeNet was pretty good but validation accuracy was less than 90%. So, I decided to amanden it.
* I tryed dfferent approaches: to include/exclude colors, to add more layers and etc. But the model was overfitting. So, I decided to stay with LeNet architecture, but to increase nubmer of neurons in layers 3-4 times because we have more number of labels than in MNIST data (43 vs 10) and to protect model from overfitting I decided to add dropout after 2nd convolution layer with 50% keep ratio. It helped to increace accuracy to 94%. 
* Also, in the future I want to increase number of obervation by adding to the trainig sample modified images (by different angles, noice and etc.) Now i don't know how to do it: by TF functionality or with other libraries. And because label distribution is far away from perfect I want to statify development sample - I think it will help better reconize rare trafic signes.

### Test a Model on New Images

#### 1. New five German traffic signs found on the web.

Here are five German traffic signs that I found on the web. They were in a much bigger resolution so, I used TF *tf.image.resize_nearest_neighbor* function to resize them to (32, 32) format.

![alt text][image4]

#### 2. Model's predictions on the new traffic signs.

The code for making predictions on my final model is located in the tenth 17-24 cells of the Ipython notebook.

Here are the results of the prediction:

| Image                   |     Prediction            | 
|:-----------------------:|:-------------------------:| 
| Speed limit (100km/h)   | Speed limit (100km/h)     | 
| Stop Sign               | Stop Sign                 |
| Priority road           | Priority road             |
| Children crossing       | Children crossing         |
| Speed limit (60km/h)    | Speed limit (60km/h)      |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.4%

#### 3. Softmax probabilities for each prediction.

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.
As we see the model is pretty confidence with all five images.

Top 5 Probabilities for the prediciton: 

 | Image   |   Top-5 Probability      | 
 |--------:|:------------------------:|
 | 1 image | 1.   0.   0.   0.   0.   |
 | 2 image | 1.   0.   0.   0.   0.   |
 | 3 image | 1.   0.   0.   0.   0.   |
 | 4 image | 1.   0.   0.   0.   0.   |
 | 5 image | 0.82 0.07 0.06 0.02 0.01 |
 
 Top 5 labels for the prediction: 
 
 | Image   |Top-5 Pred. Lab.| 
 |--------:|:--------------:|
 | 1 image |  0  1 29 38  4 |
 | 2 image | 14 13  1 38 12 |
 | 3 image | 12 40 35 13 15 |
 | 4 image | 28 30  3 11 29 |
 | 5 image |  3  2 40 16  5 | 

 Correct labels: 0, 14, 12, 28, 3
