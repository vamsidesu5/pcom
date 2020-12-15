# Train a per-frame classifier on NTU-RGBD for RGB data (use VGG16, you can get it in pytorch)

import torchvision.models as models
for files in os.listdir("./coc/pcba1/Datasets/public/NTU_RGBD/nturgb+d_rgb"):
    print(files)
