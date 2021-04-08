# Train a per-frame classifier on NTU-RGBD for RGB data (use VGG16, you can get it in pytorch)
# Dataset Info: http://rose1.ntu.edu.sg/datasets/actionrecognition.asp
# Parsing Info: https://github.com/FesianXu/NTU_RGBD120_Parser_python

# Imports
import torchvision.models as models
import os
import torchvision

# VIDEO_PATH: path to all RGB videos for NTU-RGBD dataset
video_path = "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"
train_videos = {}  # train_videos: dictionary of label to list of video names for training
test_videos = {}  # test_videos: dictionary of label to list of video names for testing
count = 0  # COUNT: total number of videos being inputted as test or train data
# Reading through all files in VIDEO_PATH and putting all videos in the matching action label's list in dictionary

# torchvision.io.read_video("/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb/S014C002P025R001A017_rgb.avi")) 


# def vgg16Model():
#     vgg16 = models.vgg16(inputs=torchvision.io.read_video(
#         "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb/S014C002P025R001A017_rgb.avi"))  # vgg16: VGG16 model (type of CNN)
#     # model_VGG16 = VGG16(include_top = False, weights = None)
#     # model_input = Input(shape = image_shape, name = 'input_layer')
#     # output_VGG16_conv = model_VGG16(model_input)
#     # #Init of FC layers
#     # x = Flatten(name='flatten')(output_VGG16_conv)
#     # x = Dense(256, activation = 'relu', name = 'fc1')(x)
#     # output_layer = Dense(num_classes,activation='softmax',name='output_layer')(x)
#     # vgg16 = Model(inputs = model_input, outputs = output_layer)
#     # vgg16.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#     vgg16.summary()
#     return vgg16


# vgg16Model()

# subjectSet = set()
# for files in os.listdir("/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"):
#     if int(files[17:20]) <= 120 and int(files[17:20]) >= 1:
#         print(files)
#         print(int(files[17:20]))
#         print("Subject ID" + files[0:4])
#         subjectSet.add(int(files[1:4]))
#         if int(files[17:20]) in set(train_videos.keys()) and int(files[1:4]) <= 16:
#             train_videos[int(files[17:20])].append(files)
#         elif int(files[17:20]) not in set(train_videos.keys()) and int(files[1:4]) <= 16:
#             train_videos[int(files[17:20])] = [files]
#         elif int(files[17:20]) in set(test_videos.keys()) and int(files[1:4]) > 16:
#             test_videos[int(files[17:20])].append(files)
#         elif int(files[17:20]) not in set(test_videos.keys()) and int(files[1:4]) > 16:
#             test_videos[int(files[17:20])] = [files]
#         count += 1
# print(test_videos)
# print(subjectSet)


video = torchvision.io.read_video(
    "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb/S014C002P025R001A017_rgb.avi")
print(video[0].shape)
