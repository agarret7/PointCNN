import math
import data_utils
import time

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from pointcnn.core import rPointCNN
from pointcnn.util import knn_indices_func
from pointcnn.layers import MLP

x = 2

# N_neighbors, dilution, N_rep, C_out
# 8          , 1,        all  , 16 * x
# 8          , 2,        all  , 32 * x
# 8          , 4,        all  , 48 * x
# 12         , 4,        120  , 64 * x
# 12         , 6,        120  , 80 * x

# Data_dim = 3

class mnist_dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i], self.labels[i]

# C_in, C_out, D, N_neighbors, dilution, N_rep, r_indices_func, C_lifted = None, mlp_width = 2
# (a, b, c, d, e) == (C_in, C_out, N_neighbors, dilution, N_rep)
paPointCNN = lambda a,b,c,d,e: rPointCNN(a, b, 3, c, d, e, knn_indices_func)

class Classifier(nn.Module):
    
    def __init__(self):
        super(Classifier, self).__init__()

        self.pcnn = nn.Sequential(
            paPointCNN(  1,  32,  8, 1, 256),
            paPointCNN( 32,  64,  8, 2, 256),
            paPointCNN( 64,  96,  8, 4, 256),
            paPointCNN( 96, 128, 12, 4, 120),
            paPointCNN(128, 160, 12, 6, 120),
        )

        self.fcn = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128,  64),  # throw in some batch normalization
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10),  # 10 digits
        )

        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.pcnn(x)[1]  # grab features
        logits = self.fcn(x)
        logits = torch.mean(logits, dim = 1)
        log_probs = self.log_softmax(logits)
        return log_probs

model = Classifier().cuda()

num_class = 10
sample_num = 160
batch_size = 32
num_epochs = 2048
jitter = 0.01
jitter_val = 0.01

rotation_range = [0, math.pi / 18, 0, 'g']
rotation_rage_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']

data_train, label_train, data_val, label_val = data_utils.load_cls_train_val("./mnist/zips/train_files.txt", "./mnist/zips/test_files.txt")

num_train = data_train.shape[0]
point_num = data_train.shape[1]

batch_num_per_epoch = int(math.ceil(num_train / batch_size))
batch_num = batch_num_per_epoch * num_epochs

dataset = mnist_dataset(data_train, label_train)
loader = DataLoader(dataset, batch_size = batch_size)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
loss_fn = nn.NLLLoss()

for _ in range(num_epochs):
    for data, label in loader:

        data = Variable(data).cuda()
        label = Variable(label.long()).cuda()
        P = data[:,:,:3]
        F = data[:,:,3:]

        optimizer.zero_grad()

        t0 = time.time()
        out = model((P, F))

        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        
        # print(loss.data[0])
