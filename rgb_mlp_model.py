import numpy as np 
import torch
from torch import nn
import scipy
from torch import optim
import os
import torch as th
from scipy import sparse

th.cuda.set_device(2)

x_train = np.load(os.path.abspath(".") + "/" + "y_data.npy")
x_test = np.load(os.path.abspath(".") + "/" + "y_data.npy")
y_train = np.load(os.path.abspath(".") + "/" + "y_data.npy")
y_test = np.load(os.path.abspath(".") + "/" + "y_data.npy")

x_train = torch.tensor(scipy.sparse.csr_matrix.todense(x_train)).float()
x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()
y_train = torch.tensor(y_train.values)
y_test = torch.tensor(y_test.values)

# simple multi-layer perceptions network

model = nn.Sequential(nn.Linear(x_train.shape[1], 64),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(64, 19),
                      nn.LogSoftmax(dim=1))

# Define the loss
criterion = nn.NLLLoss()

# Forward pass, get our logits
logps = model(x_train)
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

    output = model.forward(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    train_loss = loss.item()
    train_losses.append(train_loss)
    
    optimizer.step()

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        log_ps = model(x_test)
        test_loss = criterion(log_ps, y_test)
        test_losses.append(test_loss)

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y_test.view(*top_class.shape)
        test_accuracy = torch.mean(equals.float())
        test_accuracies.append(test_accuracy)

    model.train()

    print("Epoch: {e+1}/{epochs}.. ")
    print("Training Loss: {train_loss:.3f}.. " )
    print("Test Loss: {test_loss:.3f}.. " )
    print("Test Accuracy: {test_accuracy:.3f}")
