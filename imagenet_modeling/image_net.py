from torchvision import models
import torch 
from torchvision import transforms
from PIL import Image

# print(dir(models)) -  prints possible models available in pytorch 
alexnet = models.alexnet(pretrained=True)
# print(alexnet) - prints layers of Alexnet 
transform = transforms.Compose([            #[1] Defining a transforrm which is a combination of transaformations to be carrried out on image 
 transforms.Resize(256),                    #[2] Resizing image to 256 x 256 
 transforms.CenterCrop(224),                #[3] Crop the image 224 x 224 about the center 
 transforms.ToTensor(),                     #[4] Convert image to a PyTorch image 
 transforms.Normalize(                      #[5] Normalize image by setting image to this mean and std. dev.
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 )])
img = Image.open("dog.jpg")
img_t = transform(img)
# print(img_t.shape) # shape: [3, 224, 224]
batch_t = torch.unsqueeze(img_t, 0) # Returns a new tensor with a dimension of size one inserted at the specified position. 
# print(batch_t.shape) # shape: [1, 3, 224, 224]
alexnet.eval()
out = alexnet(batch_t)
print(out.shape)
with open('imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]
val, index = torch.max(out, 1) # Returns the maximum value of all elements in the input tensor - first value: max, second val: index of max
# print(index)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 #calculating percentage of each class possibility using softmax
# print(percentage[index[0]])
print(classes[index[0]], percentage[index[0]].item())
_, indices = torch.sort(out, descending=True) # sorts all percentages in descending order 
[print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]] # top 5 classes predicted 
