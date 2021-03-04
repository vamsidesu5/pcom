# Train a per-frame classifier on NTU-RGBD for RGB data (use VGG16, you can get it in pytorch)
# Dataset Info: http://rose1.ntu.edu.sg/datasets/actionrecognition.asp
# Parsing Info: https://github.com/FesianXu/NTU_RGBD120_Parser_python

# Imports
import torchvision.models as models
import os
import torchvision

video_path = "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb" #VIDEO_PATH: path to all RGB videos for NTU-RGBD dataset
vgg16 = models.vgg16() # vgg16: VGG16 model (type of CNN)
videos = {} # VIDEOS: dictionary of label to list of video names
count = 0 # COUNT: total number of videos being inputted as test or train data 
# Reading through all files in VIDEO_PATH and putting all videos in the matching action label's list in dictionary

for files in os.listdir("/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"):
    if int(files[17:20]) <= 20 and int(files[17:20]) >= 1:
        print(files)
        print(int(files[17:20]))
        if int(files[17:20]) in set(videos.keys()):
            videos[int(files[17:20])].append(files)
        else:
            videos[int(files[17:20])] = [files]
        count += 1
print(len(videos.get(11)))
print(count)




# video = torchvision.io.read_video(
#     "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb/S014C002P025R001A017_rgb.avi")
# print(video[0].shape)
