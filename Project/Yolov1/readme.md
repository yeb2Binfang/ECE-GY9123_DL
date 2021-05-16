<p align="center">
<a href="https://engineering.nyu.edu/"><img src="https://user-images.githubusercontent.com/68700549/118066006-eaf92080-b36b-11eb-9116-9f8e02a79534.png" align="center" height="100"></a>
</p>

<div align="center"> 
  
## New York University

 </div>

<div align = "center">
    
 # Deep Understanding of YOLOv1 and Object Detection System Based on YOLOv1
 
#### Course  Instructor:  Dr.  Chinmay Hegde

#### Binfang Ye, Simon Wang

#### Please check [Project Report for ECE-GY 9123 Deep Learning](https://github.com/yeb2Binfang/ECE-GY9123_DL/blob/main/Project/Yolov1/ECE_GY_9123_DL_Final_Report.pdf) for more details



</div>

<div align = "center">
  
## Problem Statement

</div>

Deep learning in the computer vision field is developing at a rapid pace.  Many car industries arecombining computer vision algorithms to make intelligent vehicles such as driverless cars.  How-ever, driverless cars are not ubiquitous because it is hard to detect objects around the car correctly.Therefore, Autonomous vehicles are hard to make real-time and right decisions.  To decrease thecar accident rate and to protect human beings, a good object detection algorithm that can conveyaccurate information to humans and computers is necessary.  The algorithms play an essential partin helping driverless cars ”see” the objects.  There are many object detection algorithms such asfaster-RCNN, SSD, etc.  All of object detection algorithms require large dataset to train the modelfor gaining good predict accuracy. In this course, we try to have a deep understanding of one objectdetection algorithms among them.

YOLO (You Only Look Once) family is one of the object detection algorithms and is proposed byRedmon et al. (2016). Unlike R-CNN, YOLO is a one stage algorithm that is the main reason why it runs faster than other algorithms. In addition, it is important to know that the YOLO is treating the object detection as a regression problem which is from images pixels to bounding box and classprobabilities. In this work, we study and present a recreation of the YOLO (v1) algorithm and explain it in details. We will provide a detailed summary of our project implementations, algorithm theory, and a result analysis.

<div align = "center">
  
## Literature Survey

</div>

You Only Look Once(YOLO)is proposed by Redmon et al. (2016). YOLO is an object detection algorithm that aims to solve object detection problems as a regression problem.  The YOLOv1 network divides an image into many grid cells and uses local grid cell features to make predictions of bounding boxes. It predicts all bounding boxes for all classes in an image simultaneously.

Redmon & Farhadi (2017) improved the YOLO algorithm that can detect over 9000 different object categories in real time. The improved model *(YOLO9000)* can well balance running speed andaccuracy. Compared with faster-RCNN and SSD, *YOLO9000* is much faster while achieving high accuracy on VOC 2007 at 67 FPS. Redmon & Farhadi (2017) proposed joint training that allows the model to predict data that do not have any annotation information and they tested the model on ImageNet validation set getting 19.7 mAP. 

Redmon & Farhadi (2018) updated a new YOLO network *(darknet)* that is faster and more accurate than previous work. However, the new network is larger than the prior network.  Compared with SSD,  *YOLOv3* can reach the same accuracy but three times faster on the same testing dataset. Compared with YOLOv1, *YOLOv3* can well perform on detecting the small objects. 

Bochkovskiy et al. (2020) combined different teniques including Weighted-Residual-Connections(WRC), Cross-Stage-Partial-connections(CSP), Cross mini-Batch Normalization(CmBN),  Self-adversarial-training(SAT) and Mish-activation to achieve high detect accuracy. Bochkovskiy et al.(2020) developed an efficient and fast model that is able to use in the video. The model is much easier to train compared with the prior work.


<div align = "center">
  
## Yolov1 Algorithm

</div>

Yolov1 is an end to end algorithm which puts the resized image to the Yolov1 network and outputthe thresholded bounding boxes image. The Figure 1 shows how does the YOLOv1 process the images.

<p align="center">
<img src="https://user-images.githubusercontent.com/68700549/118399662-35a8c000-b62c-11eb-9911-dae4b5bd51c0.png" align="center" height="100">
</p>
<div align="center"> 
  
Figure 1: The overview of YOLOv1 (Redmon et al., 2016)

</div>

### Network Explanation
Yolov1 Network is large, deep and complicated.  Figure 2 shows the YOLOv1 Network Architec-ture.  In our implementation, we constructed the deep neural network with 24 convolutional layers,included 4 max-pool layers and followed 2 fully connected layers at the end.

<p align="center">
<img src="https://user-images.githubusercontent.com/68700549/118400076-1743c400-b62e-11eb-8bd2-703b4fd7ed4b.png" align="center" height="200">
</p>
<div align="center"> 
  
Figure 2: The YOLOv1 network architecture

</div>

The original paper used 4096 neurons for the fully connected layer1. However, to save training time and GPU space, we changed it to 496 neurons as shown in Figure 2. Max pooling layer is able to reduce the training dimensions and we can notice that in the network,1 × 1 convolution is used often since it can increase or decrease channels easily without using many parameters. Therefore,using 1 × 1 convolution in YOLOv1 network is able to reduce the network complexity. We utilized zero-padding to ensure a more consistent layer output shape. Inputs to this network are expected tobe RGB images of shape (448 × 448 × 3). The details of the network architecture can be foundin table 1. It’s worth mentioning that layer clusters, like Conv7 to Conv14, consist essentially of repetitions of two consecutive layers (in this case, Conv7 and Con8). The purpose of this structure is to expand the number of trainable parameters to improve the network’s overall accuracy. Besides, itcan help the network capture more features from the training data due to the training set containingmassive information. Output of this network is a tensor of shape 7×7×30. So, for the output layer shown in Figure 2, we need to reshape it to the shape we expected. The original 448 × 448 × 3 input image is divided into 7 × 7grid cells, each of which contains a 30-element array. Indices 0-19 represent 20 class conditional probabilities P(Class_i|Object). Indices 20-24 and 25-29 eachindicate a prediction bounding box’s information:

* x: x-coordinate of object detector center, normalized with respect to the width of grid cell,

* y: y-coordinate of object detector center, normalized with respect to the height of grid cell,

* w: width of object detector, normalized with respect to the image width.

* h: height of object detector, normalized with respect to the image height.

* c: confidence of object detector.

Finally, class conditional probabilities from Output[:, :, 0:19] and confidences from Output [:, :, 24] and Output [:, :, 29] will determine the probability of the predicted class existing in the grid cell and how well the predicted x, y, w and h match the object. 


### Loss Function
The final layer predictions contain class probabilities and bounding box information. We will pick the maximum probability as the final prediction. We also normalize the weight w, height h, x, and y falling between 0 and 1. As mentioned, Yolov1 reframes object detection as the regression problem. Therefore, it is easy to use sum-squared error to optimize our model. The loss function used in our model is

<p align="center">
<img src="https://user-images.githubusercontent.com/68700549/118400495-dd73bd00-b62f-11eb-9ec6-59c1bdc3de65.png" align="center" height="200">
</p>


<div align = "center">
  
## Dataset Description

</div>

Our dataset of choice is the PascalVOC dataset that contains 43,223 images. This dataset offers ahuge number of annotated images and we will focus on using its “Bounding Box” annotations. Anannotation TXT file is also provided with useful information including “class”, “x”, “y”, “w”, “h”. During our training process, we will use such information to identify if the object(s) within an image and their locations in terms of XY coordinates. We can also identify the bounding boxes throughratio W and H. There are 20 object categories in total. The number from [0,19] represents airplane,bike, bird, boat, bottle, bus, car, cat, chair, cow, table, dog, horse, motorbike, person, plant, sheep, sofa,  train, and tv differently. However, we found that the data and labels (9,958 images and labelsare matched correctly) are messy in this dataset. Therefore, we cleaned up the dataset and randomly selected 5,000, 200, and 100 images as the training, validation, and testing dataset respectively. The dataset, label information, and CSV files can be  found [here](https://drive.google.com/drive/folders/1dqW6nx5gaRX-XihvkK81Kw1xhsT0j9Sr?usp=sharing)


<div align = "center">
  
## Results

We apply the model on the training dataset, evaluation dataset, testing dataset.
</div>

<p align="center">
<img src="https://user-images.githubusercontent.com/68700549/118414654-f2723f80-b673-11eb-851f-af626ed3681b.png" align="center" height="200">
</p>

<div align="center"> 
  
Figure 3: Correct predictions on training dataset

</div>

<p align="center">
<img src="https://user-images.githubusercontent.com/68700549/118414732-4e3cc880-b674-11eb-9a38-c5e8618fab83.png" align="center" height="200">
</p>

<div align="center"> 
  
Figure 4: Correct predictions on evaluation dataset

</div>

<p align="center">
<img src="https://user-images.githubusercontent.com/68700549/118414779-8c39ec80-b674-11eb-8e71-2f8ff6a13252.png" align="center" height="200">
</p>

<div align="center"> 
  
Figure 5: Correct predictions on testing dataset

</div>

<p align="center">
<img src="https://user-images.githubusercontent.com/68700549/118414801-a83d8e00-b674-11eb-82df-83d507078bc6.png" align="center" height="200">
</p>

<div align="center"> 
  
Figure 6: Wrong detections on randomly picked images

</div>


<div align = "center">
  
## How to Use?

</div>

We save our pretrained model [here](https://drive.google.com/drive/folders/1gYuX5FztKkzlj4W6k8ohfMfsuff0MbAv?usp=sharing). We can just download all of scripts including dataset.py, loss.py, utils.py, model.py and upload it to Google Colab. Then we need to downlaod the train.csv, evalution.csv, and test .csv files to access the dataset that is mentioned in section **Dataset Description**. We also have the script for generating CSV file if you want to change the number of the training, evalution, and testing dataset. We need to use Google Colab GPU to accerlate the training speed. If you want to re-train the model, it will take about 3 hours to finish using 5,000 images.  

<div align = "center">
  
## Reference

</div>

Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao.   Yolov4:  Optimal speed andaccuracy of object detection.arXiv preprint arXiv:2004.10934, 2020.

Joseph Redmon and Ali Farhadi.   Yolo9000:  better, faster, stronger.   InProceedings of the IEEEconference on computer vision and pattern recognition, pp. 7263–7271, 2017.

Joseph  Redmon  and  Ali  Farhadi.Yolov3:    An  incremental  improvement.arXiv preprintarXiv:1804.02767, 2018.

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi.  You only look once:  Unified,real-time object detection. InProceedings of the IEEE conference on computer vision and patternrecognition, pp. 779–788, 2016

[Guide on YOLO V1 implementation](https://www.youtube.com/watch?v=n9_XyCGr-MI)
