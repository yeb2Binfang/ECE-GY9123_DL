import torch
import os
import pandas as pd 
import PIL import Image 

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        #各种文件files
        #transform不知道是干嘛用的
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform = None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        #label.txt
        label.path = os.path.join(self.label_dir, self.annotations.iloc[index,1])
        #不知道这个boxes干嘛用的
        boxes = []
        with open(label.path) as f:
            for label in f.readlines():
                #感觉这部分有问题
                class_label, x, y, width, height = {
                    #感觉有点鸡肋，应该可以改
                    float(x) if float(x)!=int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                }
                #这里顺序可能会有问题
                boxes.append([class_label, x, y, width, height])

        #image files
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        image = Image.open(img_path)
        #改成tensor就是为了下面的transform好操作
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)
        
        #convert to cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            #这个是用来表示target bounding box 在 S*S中的具体哪一格，是格的index
            # y is row, x is col                                                                                                   
            i, j = int(self.S * y), int(self.S * x)
            #这个是表示在那一个cell中的具体位置，也就是那个小数点后的数
            x_cell, y_cell = self.S * x - j, self.S * y - i
            #这个好像也不大懂
            width_cell, height_cell = (
                width * self.S,
                height * self.S
            )

            if label_matrix[i,j,20]==0:
                label_matrix[i,j,20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i,j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1
        
        return image, label_matrix



