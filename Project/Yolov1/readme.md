<div align = "center">
    <h3>
        Deep Understanding of YOLOv1 and Object Detection System Based on YOLOv1
    </h3>
</div>

<div align = "center">Binfang Ye(@by2034)</div>

<div align = "center">Simon Wang(@ssw8641)</div>

#### 1	Problem Statement

Deep learning in the computer vision field is developing at a rapid pace. Many car industries are combining computer vision algorithms to make intelligent vehicles such as driverless cars. However, driverless cars are not ubiquitous because it is hard to detect objects around the car correctly. Therefore, Autonomous vehicles are hard to make real-time and right decisions. To decrease the car accident rate and to protect human beings, a good object detection algorithm that can convey accurate information to humans and computers is necessary. YOLO (You Only Look Once) is one of the object detection algorithms and plays an essential part in helping driverless cars "see" the objects. In this work, we present a recreation of the YOLO (v1) algorithm. 

#### 2	Literature Survey

In 2015, Redmon, et al. introduced *You Only Look Once*(YOLO) object detection algorithm that aims to solve object detection problems as a regression problem. A YOLO (v1) network divides an image into many grid cells and uses local grid cell features to make predictions of bounding boxes. It predicts all bounding boxes for all classes in an image simultaneously. 

#### 3 Dataset Description

Our dataset of choice is Pascal_VOC dataset that contains 43,223 images. This dataset offers a huge number of annotated images and we will focus on using its “Bounding Box” annotations. An annotation TXT file is also provided with useful information including “class”, “x”, “y”, “w”, “h”. During our training process, we will use such information to identify the object(s) within an image and their locations in terms of XY coordinates. We can also identify the bounding boxes through ratio W and H.

#### 4	YOLOv1 Network

Yolov1 Network is large and complicative. Figure 1 shows the YOLOv1 Network Architecture

<img src="https://user-images.githubusercontent.com/68700549/112764238-96752e00-8fd5-11eb-8ef3-57f8750d6bbe.png" alt="WeChat Screenshot_20210328145449" style="zoom:50%;" />

<div align=center><B>Figure 1.</B> YOLOv1 net<div>

We recreate the Yolov1 Network. Figure 2 shows the details of the network. 

![pasted image 0](https://user-images.githubusercontent.com/68700549/115912317-296e8000-a43d-11eb-8943-d9e77fa9b8bc.png)

<div align = "center"><B>Figure 2.</B> Yolov1 network details</div>

##### 4.1 Network Explanation

In our implementation, we constructed our deep neural network with 24 convolutional layers, 4 max-pool layers and 2 fully connected layers. We utilized zero-padding to ensure a more consistent layer output shape. Inputs to this network are expected to be RGB images of shape ($448*448*3$). The network’s detailed architecture can be found in Figure 2. It’s worth mentioning that layer clusters, like Conv7 to Conv14, consist essentially of repetitions of two consecutive layers (in this case, Conv7 and Con8). 

The purpose of this structure is to expand the number of trainable parameters to improve the network's overall accuracy.
Output of this network is a tensor of shape 7730. The original 448448 input image is divided into 77 grid cells, each of which contains a 30-element array. Indices 0-19 represent 20 class conditional probabilities $P(Class_i|Object)$. Indices 20-24 and 25-29 each indicate a prediction bounding box’s information: 

​	x: x-coordinate of object center, normalized with respect to the width of grid cell,

​	y: y-coordinate of object center, normalized with respect to the height of grid cell,

​	w: width of object, normalized with respect to the image width,h: height of object, normalized with respect to the image height,

​	c: confidence of object.

Finally, class conditional probabilities from Output[:, :, 0:19] and confidences from Output [:, :, 24] and Output [:, :, 29] will determine the probability of the predicted class existing in the grid cell and how well the predicted x, y, w and h fit the object.

##### 4.2 Loss function

As mentioned, Yolov1 reframes object detection as the regression problem. Therefore, it is easy to use sum-squared error to optimize our model. The loss function used in our model is

![WeChat Screenshot_20210428111729](https://user-images.githubusercontent.com/68700549/116428930-62c73700-a813-11eb-9b17-38e293673313.png)


Part$(I)$ and part$(II)$ is penalizing the coordinate error and the ratio error. For part$(II)$, the sum-squared error will equally weight errors in large boxes and small boxes if we do not add square root on $w$ and $h$. The small boxes should matter more than the large boxes when we use the same deviations on them. Taking the square root can partially address this issue. Figure 2 shows how does the square root work. Part$(III)$ and part$(IV)$ will penalize the confident errors. But for part$(III)$, we take the bounding box that is responsible for the predicting object, while for part$(IV)$, we take the bounding box that is not responsible for the predicting object. In the loss function, $1_{ij}^{obj}$ denotes that the $j^{th}$ bounding box predictor in cell $i$ takes the responsible for the prediction. We will only take only one bounding box that has the highest IOU with the ground truth in each cell to be responsible for predicting object. We use two parameters $\lambda _{cood}=5$ and $\lambda _{noobj}=0.5$ to make the model stable. Many grid cells in a image probably do not have many objects so that the confident scores for those cells are 0, which will affect the gradient from cells that do have the object. Therefore, we increase the loss for cells that do contain the object and decrease the confident loss from cells that do not contain objects. Part$(V)$ will penalize the classification error. $1_i^{obj}$ denotes that whether an object appears in the cell it or not.

<img src="https://user-images.githubusercontent.com/68700549/115788467-a3492f80-a391-11eb-87e9-90e5016e1fe1.png" alt="WeChat Screenshot_20210422173843" style="zoom: 67%;" />

**Figure 2**: It shows that the small boxes matter more than the large boxes if we add square root in the loss function.



#### 5	Preliminary Results

As of April 22nd, we have successfully implemented model architecture, training procedure and loss function of our network. These files can be found under the Github repository: https://github.com/yeb2Binfang/ECE-GY9123_DL/tree/main/Project/Yolov1, in files model.py, train.py and loss.py, respectively.

#### 6	Remaining Work and Possible Challenges

The next step we will do is that we will train our network on NYU HPC and apply the trained model to the real world pictures. We will compare with the other object detection algorithms if it is possible. Our current possible challenge is primarily our computational power. Since we need to train a convolutional neural network with over 20 layers with a 4G training data. It is also difficult for us to estimate what sort of GPU we need and its resulting training time in our scenario. 

#### 7	References
https://www.kaggle.com/dataset/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2	

​			PASCAL_VOC Dataset	
https://arxiv.org/pdf/1506.02640.pdf	

​			You Only Look Once: Unified, Real-Time Object Detection

​			Joseph Redmon, Santosh Divvala, Ross Girshick , Ali Farhadi
https://github.com/abeardear/pytorch-YOLO-v1	

​			Guide on YOLO V1 implementation
