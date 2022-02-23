
import numpy as np
import os
from tempfile import TemporaryFile

directory = '/coc/scratch/dscarafoni3/50salads/mstcn_data/50salads/features'
x_train = np.empty((0,2048),int)
y_train = np.empty((0,1),int)
x_train_data = TemporaryFile()
y_train_data = TemporaryFile()
rgb_video = np.load(directory + "/" + "rgb-01-1.npy")
print(rgb_video.shape)

counter = 0
for filename in os.listdir(directory):
    #For debgugging purposes
    print(counter)
    counter += 1
    print(filename)
    #Actual Code
    rgb_video = np.load(directory + "/" + filename)
    rgb_video_transposed = np.transpose(rgb_video)
    print(rgb_video_transposed.shape)
    x_train = np.append(x_train,rgb_video_transposed,axis=0)
    print(x_train.shape)
np.save(x_train_data, x_train)
np.save(y_train_data, y_train)
print(x_train)


