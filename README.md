# **Term1 Project2: Traffic Sign Recognition Classifier**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---
## **Summary**
* The project builds a convolutional neural network (CNN) to classify German traffic lights, achieved test accuracy **95%.**
* Best accruacy is achieved by preprocessing image with **CLAHE equalization**, and then through a CNN model, based on LeNet with adding **Batch Normalization, Dropout, and more hidden convolution layer and full connected layers**. All the parameters are tuned with iterative approach.
* **Data augmentation** is implemented with 4x more samples with geometry transformation for training set, but final performance is slightly worse.  
* Final model is tested on new web images. **Grayscale** and RGB accuracy are comparable in training, validation and test sets with final model, but **grayscale is more reliable with new web images**.
* Grayscale version is *Traffic_Sign_Classifier_grayscale.ipynb (.html)*. RGB version is *Traffic_Sign_Classifier_color.ipynb(.html)*
---
## **Steps for Building a Traffic Sign Recognition Classifier**

The steps of this project are the following:
1. Load the data set, explore, summarize and visualize the data set
* Improve the image quality with cv2 CLAHE equalizer colored and grayscale
* Augment images with 3 types of geometry transformations
* Design, train and test a modified model architecture with hyperparameters tuning
* Make predictions on new images with top5 softmax probabilities
---
### **Step1: Data Set Summary & Exploration**

#### A. basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### B. Include an exploratory visualization of the dataset.

From CSV file, there are total 43 classes:

|Class ID |Sign Names|
|-------|--------|
|0|	Speed limit (20km/h)|
|1|	Speed limit (30km/h)|
|2|	Speed limit (50km/h)|
|3|	Speed limit (60km/h)|
|4|	Speed limit (70km/h)|
|5|	Speed limit (80km/h)|
|...|...|
|38|	Keep right|
|39|	Keep left|
|40|	Roundabout mandatory|
|41|	End of no passing|
|42|	End of no passing by vehicles over 3.5 metric tons|

Here is an exploratory visualization of the data set. It is a bar chart showing how the dataset distributed.

**The distribution of training, validation and test data is very similar. That is great. However, the distribution is NOT uniform.**

![alt text](https://github.com/zoespot/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/distribute.PNG)

**From plotting the random selected original training data, a lot of images looks quite dark.**

![alt text](https://github.com/zoespot/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/selected_original.PNG)

**As an example, below is the one image with its respective class number. This is #33: "Turn right ahead" **

![alt text](https://github.com/zoespot/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/random_image_classnum.PNG)

---
### **Step2: Image Quality Improvement**
One way to improve image quality is to improve the contrast of the images with OpenCV histogram equalization (https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html).

**CLAHE** (contrast limited adaptive histogram equalization) is an advanced version with equalizing by individual region.

With color plot (top: original, bottom: CLAHE with gridsize=4 cliplimit=6):

![alt text](https://github.com/zoespot/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/color_clahe.PNG)

With grayscale plot (top: original, bottom: CLAHE with gridsize=4 cliplimit=12):

![alt text](https://github.com/zoespot/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/grayscale_clahe.PNG)

**Grayscale generates clearer image than color image through CLAHE, which is also proved by training with CNN models with different CLAHE parameters. Grayscale and CLAHE gridsize=4 cliplimit=12 is used in final design.**

When plotting the improved images of the original selected training data, the dark images are more visible. So are the valid dataset images.

![alt text](https://github.com/zoespot/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/selected_clahe_gray.PNG)

---
### **Step3: Images Augmentation**

Image augmentation can be used to increase the size of dataset. It can help increase the training dataset size, which may help the CNN works better.

 OpenCV geometry augmentation methods are available in  (https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html).

With **Rotation**, **Affine transformation** and **Perspective transformation** applied, the training set size increases by **4 times** from **34799** to **139196**. Iamges numbers in each class increase by 4 times proportionally.  

Examples of augmented images are:

![alt text](https://github.com/zoespot/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/selected_augmented_gray.PNG)

Both colored and grayscale versions are tested in modified CNN models below.

---
### **Step4: CNN Model Architecture**

#### 1. Preprocessing
Preprocessing images to grayscale and CLAHE equalization is described in Step2 and data augmentation is described in Step3.
I tried both colored and grayscale, and found grayscale generates model with higher accuracy, so final model architecture below uses grayscale.

#### 2. Model Architecture
My final model architecture is listed below. **Convolution layer3a is parallel to Convolution layer3, and their output flatten layers are concatenated to Full connected Layer1. **

 (Reference:http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

![alt text](https://github.com/zoespot/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/Lecun_model.PNG)

My final model consists of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image (preprocessed of 32x32x3 RGB image with CLAHE equalization)   							|
| Convolution Layer1 5x5| 1x1 stride, valid padding, outputs 28x28x6	|
|Batch normalization Layer1| scale =1, variance_epsilon =0.001|
| RELU			|												|
| Max pooling	 Layer1     	| 2x2 stride, outputs 14x14x6 				|
| Convolution Layer2 5x5|1x1 stride, valid padding, outputs 10x10x16|
|Batch normalization Layer2| scale =1, variance_epsilon =0.001|
| RELU			|												|
| Max pooling	 Layer2     	| 2x2 stride, outputs 5x5x16 				|
| Convolution Layer3 2x2|1x1 stride, valid padding, outputs 4x4x32|
|Batch normalization Layer3| scale =1, variance_epsilon =0.001|
| RELU			|												|
| Max pooling	Layer3      	| 2x2 stride, outputs 2x2x32 				|
| Convolution Layer3a 3x3| parallel applied to output of Convolution Layer2,  1x1 stride, valid padding, outputs 8x8x32|
| full connected	Layer1	| concatenated of flatten layer of output of "Max pooling Layer3" and flatten layer of output of "Convolution Layer3a", Input 1024 Output 400|
| RELU			|												|
| Full connected	Layer2				| Input 400 Output 120   									|
| RELU			|												|
|Dropout fc2						|			Keep prob = 0.5									|
| Full connected	Layer3				| Input 120 Output 84   									|
| RELU			|												|
|Dropout fc3						|			Keep prob = 0.4									|
| Full connected	Layer4				| Input 84 Output 43|
|Softmax| softmax probability                |

 #### 3. Optimizer Hyperparameters
 Adam optimizer is used in the model. Adaptive Moment Estimation (Adam) optimizer is chosen as it computes adaptive learning rates for each parameter, thus quick to converge. (Ref: http://ruder.io/optimizing-gradient-descent/index.html#adam)
 
 The final model hyperparameters are listed as below:

 |Optimizer| Adam|Note|
 |---|---|---|
 |Batchsize|512|Smaller batchsize (32,16) gives higher accuracy in small epochs (10), larger batchsize (256, 512) gives slightly higher accuracy with large Epochs (400)|
 |Epochs|550||
 |Learning Rate|0.0001|usually 0.001, 0.0003 with 200 Epochs can already gives 93% accuracy|

Regarding to accurancy variations with different compbinations of Batchsize and Epochs, the following figure illustrates several runs I have taken: 

![alt text](https://github.com/zoespot/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/accuracy_bz_epochs.PNG)

When Epoch number is small, smaller batch size is better, because large batch size tends to converge too quickly to local minimums. (Ref: https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network). When Epoches increases, batchsize matters less since more epochs help the optimizer jumping out of local minimums as well. Also with more epochs, slightly better results is observed with larger batchsize. It might come from the quicker convergence with larger batchsize. If given more epochs with smaller batchsize, it will converge to similar level of accraucy. 

#### 4. Approach to reach final model and hyperparameters

With final model and hyperparameters listed above,
* Training set accuracy is 95%
* **Validation set accuracy is 95%**
* **Test set accuracy is 95%**

The approach I tried to reach the final model is trying each of following ideas by steps to check if accuracy gets better:

* Use image with **histogram equalization** and **data augmentation**, both useful, but **CLAHE equalization** is even better with tuned parameters
* Add **Batch  Normalization** and **Dropout** layers to the model
* Experiment with **Dropout keep probability** in different layers, found only in **the last one or two full connected layer is most effective**, 0.85 is used when epochs=10-20, 0.5/0.4 is used when epochs>100
* Add additional **parallel convolution layer** and **one more full connected layer** to the model, tune the layers' sizes
* Increase **Epochs**, **Batchsize**, decrease **Learning Rate**
* Final model with **data augmentation** is worse than with original improved images. So data augmentation is **not used**.
* Final model with **Grayscale** is better than with RGB Images, so grayscale is used.

Several key trials with different model architecture and hyperparameters with their accuracies are listed below:

|Architecture|Data|hyperparameters|Training accuracy|Validation accuracy|Note|
|------------|----|---------------|-----------------|-------------------|----|
|LeNet|**Original**|Epochs=10 Batchsize=128 training rate =0.001|97.7%|84.9%|Overfitting|
|LeNet|**Histogram Equalization**|Epochs=10 Batchsize=128 training rate =0.001|96.2%|86.4%|Improved with image equalization|        
|LeNet|**Augmented Image** Histogram Equalization|Epochs=10 Batchsize=128 training rate =0.001|96.8%|89%|Improved with 4x more training data|
|LeNet, **Batch normalization (BN), Dropout**|Histogram Equalization|Dropout_fc2 =0.75 Epochs=10 Batchsize=128 training rate =0.001|92.4%|90.5%|Dropout is useful, only last layer fc2 dropout needed|
|LeNet, BN, Dropout| **CLAHE Equalization**|Dropout_fc2 =0.75 Epochs=10 Batchsize=128 training rate =0.001|98.8%|91.1%|CLAHE is better than histogram equalization |
|LeNet, BN, Dropout, **Add one more full connected layer (fc3)**|CLAHE Equalization|Dropout_fc3 =0.85 Epochs=10 Batchsize=128 training rate =0.001|99.4%|92.2%|one more full connected layer is useful|
|LeNet, BN, Dropout, Add fc3, **Parallel conv layer (add convolution layer 3a in parallel, fc1 layer concatenated)**, tune fc layers dimensions |CLAHE Equalization|Dropout_fc3 =0.85 Epochs=30 Batchsize=128 training rate =0.0003|100%|93.6%|tune fc layer sizes to 1024->400->120->84->34|
|LeNet, BN, Dropout, Add fc3, Parallel conv layer|**Grayscale**, CLAHE Equalization  |**Dropout_fc2=0.5, Dropout_fc3 =0.4, Epochs=200** Batchsize=256 training rate =0.0003|99.3%|95.2%|95.1% Tune CLAHE equalizer parameters to gridsize= 4 cliplimit=12, increase epochs, batchsize|
|LeNet, BN, Dropout, Add fc3, Parallel conv layer|Grayscale, **w/wo Image Augmentation**, CLAHE Equalization  |Dropout_fc2=0.5, Dropout_fc3 =0.4, Epochs=450 Batchsize=512 training rate =0.0001|99.5% (with Image Augmentation), 99.1% (Without Image Augmentation)|**93.9% (with Image Augmentation), 94.8% (Without Image Augmentation)**|Image Augmentation 4x doesn't help|
|LeNet, BN, Dropout, Add fc3, Parallel conv layer|**Grayscale/RGB**, CLAHE Equalization  |Dropout_fc2=0.5, Dropout_fc3 =0.4, Epochs=300 Batchsize=512 training rate =0.0003|99.3% (Grayscale), 99.4% (RGB)|**96% (Grayscale), 95.2% (RGB)**|Test Accuracy 94.9%(Grayscale) 94.9%(RGB), **grayscale and RGB accuracy are comparable in test sets, but grayscale is more reliable with new web images**|

#### An iterative approach was chosen:
* LeNet is the first architecture that was tried. It's chosen because it's powerful CNN model and easy to construct.
* The problems with the initial architecture is mainly **overfitting**. Accuracy of training set is 8% more than the validation set.
* The architecture is adjusted by adding **batch normalization** and **dropout**, and then **adding more hidden layers** to improve the overfitting issue.
* Layer sizes (both convolution and full connected layer) are tuned. Dropout keep probability value and which layer to apply dropout is also tuned. CLAHE equalization gridsize and cliplimit are tuned too. Dropout is decreased when epochs numbers are large, because it can further eliminate overfitting. Surprisingly to see, dropout is more effective when only applied to last one or two layers.
* For this German traffic sign application, the image resolution is very low 32x32. If our dataset includes higher resolution quality images, the model will work much better.
* To prove the model works well, we want to see final model's accuracy on the training, validation and test set are all high and close to each other. Obviously, the quality of validation set is worse than training and test set in this example.
---
### **Step5: Predictions On New Images with Softmax Probabilities**

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web with their classes on the 1st row, and grayscale equalized image with predicted classes on the 2nd row:

![alt text](https://github.com/zoespot/CarND-Traffic-Sign-Classifier-Project/blob/master/Images/prediction.PNG)

The second image is classified incorrectly. Others are correct. That's probably because the resolution is quite low and it has more features in it compared with other images.  

Test accuracy for these new 5 images is 80%.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Yield					| Yield											|
| Road work    			| Slippery road									|
| Stop Sign      		| Stop sign   									|
| 70 km/h	      		| 70 km/h						 				|
|Turn right ahead		| Turn right ahead	   							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares unfavorably to the accuracy on the test set of 95%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

**1st image: Yield.** The top five soft max probabilities were [13, 36, 15, 41,  2]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .9999         			| Yield  									|
| 5e-6     				| Go straight or right 										|
| 1.6e-6					|No vehicles										|
| 2e-7	      			|End of no passing					 				|
| 5e-8				    | Speed limit (50km/h)   							|

**2nd image: Road work.** The top five soft max probabilities were [30, 23, 11, 19,  6]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .25         			| Beware of ice/snow  									|
| .24   				| Slippery road 										|
|.14					|Right-of-way at the next intersection										|
| .14      			|Dangerous curve to the left					 				|
|.06			    | End of speed limit (80km/h)					|


**3rd image: stop sign.** The top five soft max probabilities were [14, 15, 36, 38, 18]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .9999         			| Stop sign   									|
| 5e-6     				| No vehicles										|
| 8e-9					|Go straight or right									|
| 2e-10	      			|Keep right					 				|
| 4e-13				    | General caution   							|

**4th image: Speed limit (70km/h) .** The top five soft max probabilities were [ 4,  0,  1, 26,  8]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .38         			| Speed limit (70km/h) 									|
| .33     				| Speed limit (20km/h)									|
| .20					|Speed limit (30km/h)									|
| .03	      			|Traffic signals				 				|
| .02				    | Speed limit (120km/h)  							|

**5th image: turn right ahead.** The top five soft max probabilities were [33, 14, 42, 15, 25]

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .73         			| turn right ahead  									|
| .16     				| Stop sign 										|
| .03					|End of no passing by vehicles over 3.5 metric tons										|
| .02	      			|No vehicles				 				|
| .015				    |Road work   							|

**Final thoughts:**
* Grayscale and RGB accuracy are comparable in test sets, but grayscale is more reliable with new web images.
* Training, validation and test sets distribution is highly unbalanced. That might be one reason why the web images accuracy is much less than test set.
