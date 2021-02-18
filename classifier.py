# Train a per-frame classifier on NTU-RGBD for RGB data (use VGG16, you can get it in pytorch)
# Dataset Info: http://rose1.ntu.edu.sg/datasets/actionrecognition.asp
# Imports 
import torchvision.models as models
import os
import torchvision

video_path = "/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"
# reader = torchvision.io.VideoReader(video_path, "S014C002P025R001A017_rgb.avi")
# reader_md = reader.get_metadata()
# print(reader_md["video"]["fps"])

# for files in os.listdir("/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"):
#     print(files)
video = torchvision.io.read_video("/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb/S014C002P025R001A017_rgb.avi")
print(video[0].shape)