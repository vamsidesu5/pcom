import numpy as np 
import torch
from torch import nn
import scipy
from torch import optim
from scipy import sparse
import os
import torch as th

th.cuda.set_device(2)

x_train = np.load(os.path.abspath(".") + "/" + "x_train.npy")
x_test = np.load(os.path.abspath(".") + "/" + "x_test.npy")
y_train = np.load(os.path.abspath(".") + "/" + "y_train.npy")
y_test = np.load(os.path.abspath(".") + "/" + "y_test.npy")

# x_train = torch.tensor(scipy.sparse.csr_matrix.todense(x_train)).float()
# x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

print(x_train.dtype)
print(x_test.dtype)


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

epochs = 50
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