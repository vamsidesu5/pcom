# Train a per-frame classifier on NTU-RGBD for RGB data (use VGG16, you can get it in pytorch)
import torchvision.models as models
import os
import torchvision
# for files in os.listdir("/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"):
#     print(files)
torchvision.io.read_video("/coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb/S014C002P025R001A017_rgb.avi")