# Train a per-frame classifier on NTU-RGBD for RGB data (use VGG16, you can get it in pytorch)
# Dataset Info: http://rose1.ntu.edu.sg/datasets/actionrecognition.asp
# Parsing Info: https://github.com/FesianXu/NTU_RGBD120_Parser_python

# Imports
import torchvision.models as models
import os
import torchvision

video_path = "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"
vgg16 = models.vgg16()
# reader = torchvision.io.VideoReader(video_path, "S014C002P025R001A017_rgb.avi")
# reader_md = reader.get_metadata()
# print(reader_md["video"]["fps"])

count = 0
for files in os.listdir("/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"):
    if int(files[17:20]) <= 20 & int(files[17:20]) >= 1:
        print(files)
        count += 1
print(count)
video = torchvision.io.read_video(
    "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb/S014C002P025R001A017_rgb.avi")
print(video[0].shape)
