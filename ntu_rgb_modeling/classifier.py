# Train a per-frame classifier on NTU-RGBD for RGB data (use VGG16, you can get it in pytorch)
# Dataset Info: http://rose1.ntu.edu.sg/datasets/actionrecognition.asp
# Parsing Info: https://github.com/FesianXu/NTU_RGBD120_Parser_python

# Imports
import torchvision.models as models
import os
import torchvision
import torch
from torchvision import transforms

# VIDEO_PATH: path to all RGB videos for NTU-RGBD dataset
video_path = "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"
train_videos = {}  # train_videos: dictionary of label to list of video names for training
test_videos = {}  # test_videos: dictionary of label to list of video names for testing
count = 0  # COUNT: total number of videos being inputted as test or train data
# Reading through all files in VIDEO_PATH and putting all videos in the matching action label's list in dictionary

# torchvision.io.read_video("/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb/S014C002P025R001A017_rgb.avi")) 


def vgg16Model():
    vgg16 = models.vgg16(inputs=torchvision.io.read_video(
        "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb/S014C002P025R001A017_rgb.avi"))  # vgg16: VGG16 model (type of CNN)
    # model_VGG16 = VGG16(include_top = False, weights = None)
    # model_input = Input(shape = image_shape, name = 'input_layer')
    # output_VGG16_conv = model_VGG16(model_input)
    # #Init of FC layers
    # x = Flatten(name='flatten')(output_VGG16_conv)
    # x = Dense(256, activation = 'relu', name = 'fc1')(x)
    # output_layer = Dense(num_classes,activation='softmax',name='output_layer')(x)
    # vgg16 = Model(inputs = model_input, outputs = output_layer)
    # vgg16.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    vgg16.summary()
    return vgg16



def train_test_split(): 
    subjectSet = set()
    for files in os.listdir("/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"):
        if int(files[17:20]) <= 120 and int(files[17:20]) >= 1:
            print(files)
            print(int(files[17:20]))
            print("Subject ID" + files[0:4])
            subjectSet.add(int(files[1:4]))
            if int(files[17:20]) in set(train_videos.keys()) and int(files[1:4]) <= 16:
                train_videos[int(files[17:20])].append(files)
            elif int(files[17:20]) not in set(train_videos.keys()) and int(files[1:4]) <= 16:
                train_videos[int(files[17:20])] = [files]
            elif int(files[17:20]) in set(test_videos.keys()) and int(files[1:4]) > 16:
                test_videos[int(files[17:20])].append(files)
            elif int(files[17:20]) not in set(test_videos.keys()) and int(files[1:4]) > 16:
                test_videos[int(files[17:20])] = [files]
            count += 1
    print(test_videos)
    print(subjectSet)


video = torchvision.io.read_video(
    "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb/S014C002P025R001A017_rgb.avi")
print(video[0].shape)
video_tensor = video[0]
video_tensor = video_tensor.permute(0,3,1,2) # to swap dimensions 
print(video_tensor.shape)

# transform = transforms.Compose([            #[1] Defining a transforrm which is a combination of transaformations to be carrried out on image 
#  transforms.Resize(256),                    #[2] Resizing image to 256 x 256 
#  transforms.CenterCrop(224),             #[3] Crop the image 224 x 224 about the center                      
#  transforms.Normalize(                      #[5] Normalize image by setting image to this mean and std. dev.
#  mean=[0.485, 0.456, 0.406],                
#  std=[0.229, 0.224, 0.225]                  
#  )])

# video_t = transform(video_tensor)
# print(video_t.shape)





# # # Preprocessing 
# what is Dataset class in script code and how can i use it? What does dataset hold?
# I have a tensor of 183 framess...how do unpack each one in the tensor and then "apply" a label?
# Can I store all this tensor data somewhere since this takes a while to get tensors for all  120,000 videos?
# Once, I have applied the labels for each tensor, how do I pass in the train and labels into the model?

# # # Model 
# Can I use the same model strcuture for VGG?
# What is Trainer at the end?
