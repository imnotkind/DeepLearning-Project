import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# cuda setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = torch.device('cuda')

# build model
class Net(nn.Module):
    def __init__(self, input_shape=(6, 8, 8), output_shape=1,
                 filter_size=3, num_filters=(32, 128, 64)):
        super(Net, self).__init__()
        C, H, W = input_shape
        F1, FC1, FC2 = num_filters
        
        self.conv = nn.Conv2d(C,  F1, filter_size)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(F1 * filter_size * filter_size, FC1)
        self.fc2 = nn.Linear(FC1, FC2)
        self.fc3 = nn.Linear(FC2, output_shape)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class FCNet(nn.Module):
    def __init__(self, input_shape=(6, 8, 8), output_shape=1,
                num_layers=(600, 400, 200)):
        super(FCNet, self).__init__()
        C, H, W = input_shape
        FC1, FC1, FC2 = num_layers

        self.fc1 = nn.Linear(C * H * W, FC1)
        self.fc2 = nn.Linear(FC1, FC2)
        self.fc3 = nn.Linear(FC2, output_shape)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    # cuda setting
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda = torch.device('cuda')

    # I/O shape
    input_size = (6, 8, 8)
    output_size = 1

    # hyperparameters
    epoches = 200

    # dummy dataset
    class RandomDataset(Dataset):
        def __init__(self, input_size, length):
            size = np.prod(input_size)
            
            self.input_size = input_size
            self.len = length
            self.data = torch.randn(length, size)

        def __getitem__(self, index):
            inputs = torch.reshape(self.data[index], shape=self.input_size)
            label = torch.rand(1)
            return inputs, label

        def __len__(self):
            return self.len

    dataset = RandomDataset(input_size, 100)
    rand_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    
    net = Net().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # train
    for epoch in range(epoches):
        running_loss = 0.0
        for i, data in enumerate(rand_loader):
            inputs, labels = data
            inputs = Variable(inputs.cuda(), requires_grad=True)
            labels = Variable(labels.cuda(), requires_grad=True)
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        if epoch % 10 == 9:
            print('[%03d/%d] loss: %.3f' % (epoch+1, epoches, running_loss / (i + 1)))

    # evaluate test RMSE
    testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    loss = 0.0
    num_test = 0
    for data in testloader:
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        outputs = net(inputs)
        loss += torch.sum((labels - outputs) ** 2)
        num_test += len(labels)
    print("test RMES: %.6f" % torch.sqrt(loss / num_test).item())
