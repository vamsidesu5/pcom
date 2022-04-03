import numpy as np
import torch
from torch import nn
import scipy
from torch import optim
from scipy import sparse
import os
import torch as th
import pickle
from sklearn.model_selection import train_test_split

th.cuda.set_device(2)

# path_to_data = /coc/pcba1/schaganti6/rgb_data/
infile = open("/coc/pcba1/schaganti6/rgb_data/rgb_to_imu_proj_set.pkl",'rb')
dict = pickle.load(infile)
train_rgb = dict["train_set"][0]
train_imu = dict["train_set"][1]
train_rgb = train_rgb.double()
train_imu = train_imu.double()
train_rgb_imu = th.cat((train_rgb, train_imu),1)

test_rgb = dict["test_set"][0].double()
test_imu = dict["test_set"][1].double()
eval_rgb = dict["val_set"][0].double()
eval_imu = dict["val_set"][1].double()

full_test_rgb = th.cat((test_rgb, eval_rgb),0)
full_test_imu = th.cat((test_imu, eval_imu),0)
full_test_rgb_imu = th.cat((full_test_rgb,full_test_imu),1)

full_data = th.cat((train_rgb_imu,full_test_rgb_imu),0)

print(full_data.size())

# x_train = np.load(os.path.abspath(".") + "/" + "x_train.npy")
# x_test = np.load(os.path.abspath(".") + "/" + "x_test.npy")
# y_train = np.load(os.path.abspath(".") + "/" + "y_train.npy")
# y_test = np.load(os.path.abspath(".") + "/" + "y_test.npy")
y_data = np.load(os.path.abspath(".") + "/" + "y_data.npy")
# y_data = torch.from_numpy(y_data)
# full_dataset = th.cat((full_data,y_data),1)

x_train, x_test, y_train, y_test = train_test_split(
    full_data.numpy(), y_data, test_size=0.2, random_state=42, shuffle=True, stratify=y_data)

print(type(x_train))

# x_train = torch.tensor(scipy.sparse.csr_matrix.todense(x_train)).float()
# x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
#
#
# simple multi-layer perceptions network

model = nn.Sequential(nn.Linear(x_train.shape[1], 64),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(64, 19),
                      nn.LogSoftmax(dim=1)
)

# Define the loss
criterion = nn.NLLLoss()

# Forward pass, get our logits
logps = model(x_train.float())
# Calculate the loss with the logits and the labels
loss = criterion(logps, y_train)

loss.backward()

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.Adam(model.parameters(), lr=0.002)


train_losses = []
test_losses = []
test_accuracies = []

epochs = 100
for e in range(epochs):
    optimizer.zero_grad()

    output = model.forward(x_train.float())
    loss = criterion(output, y_train)
    loss.backward()
    train_loss = loss.item()
    train_losses.append(train_loss)

    optimizer.step()

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        log_ps = model(x_test.float())
        test_loss = criterion(log_ps, y_test)
        test_losses.append(test_loss)

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y_test.view(*top_class.shape)
        test_accuracy = torch.mean(equals.float())
        test_accuracies.append(test_accuracy)

    model.train()

    print("Epoch "  + str(e+1))
    print("Training Loss: " ,train_loss)
    print("Test Loss: " , test_loss.item())
    print("Test Acc: " , test_accuracy.item())
