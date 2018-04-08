"""
I got tired of cleaning the code base, so this file will stay as is
probably, unless I really want it to be cleaner.
"""

import os

import math
import random
import data_utils
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from PointCNN import RandPointCNN
from PointCNN import knn_indices_func_gpu
from PointCNN.core.util_layers import Dense

from PointCNN.mnist.visualize import make_dot

random.seed(0)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# C_in, C_out, D, N_neighbors, dilution, N_rep, r_indices_func, C_lifted = None, mlp_width = 2
# (a, b, c, d, e) == (C_in, C_out, N_neighbors, dilution, N_rep)
# Abbreviated PointCNN constructor.
AbbPointCNN = lambda a,b,c,d,e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)

class mnist_dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.pcnn1 = AbbPointCNN(  1,  32,  8, 1,  -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN( 32,  64,  8, 2,  -1),
            AbbPointCNN( 64,  96,  8, 4,  -1),
            AbbPointCNN( 96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128,  64, drop_rate = 0.5),
            Dense( 64,  10, with_bn = False, activation = None)
        )

    def forward(self, x):
        x = self.pcnn1(x)
        if False:
            print("Making graph...")
            k = make_dot(x[1])

            print("Viewing...")
            k.view()
            print("DONE")

            assert False
        x = self.pcnn2(x)[1]  # grab features

        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim = 1)
        return logits_mean

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

training_set = mnist_dataset(data_train, label_train)
training_loader = DataLoader(training_set, batch_size = batch_size)

testing_batch_size = 256
testing_set = mnist_dataset(data_val, label_val)
testing_loader = DataLoader(testing_set, batch_size = testing_batch_size)

lr = 0.01
decay_steps = 8000
decay_rate = 0.6
lr_min = 0.00001

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
loss_fn = nn.CrossEntropyLoss()

global_step = 1

model_save_dir = os.path.join(CURRENT_DIR, "models", "mnist2")
os.makedirs(model_save_dir, exist_ok = True)

losses = []
accuracies = []

if False:
    latest_model = sorted(os.listdir(model_save_dir))[-1]
    model.load_state_dict(torch.load(os.path.join(model_save_dir, latest_model)))

for e in range(1, num_epochs + 1):

    print("EPOCH %i of %i" % (e, num_epochs))

    m_loc = os.path.join(model_save_dir, "save_e%.4d" % e)
    torch.save(model.state_dict(), m_loc)
    np.savez_compressed(os.path.join(CURRENT_DIR, "losses"), losses)
    np.savez_compressed(os.path.join(CURRENT_DIR, "accuracies"), accuracies)

    # Decaying learning rate
    if e > 1:
        lr *= decay_rate ** (global_step // decay_steps)
        if lr > lr_min:
            print("NEW LEARNING RATE:", lr)
            optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)

    for data, label in training_loader:

        model = model.train()

        label = label.long()
        P = data[:,:,:3]
        F = data[:,:,3:]

        offset = int(random.gauss(0, sample_num // 8))
        offset = max(offset, -sample_num // 4)
        offset = min(offset, sample_num // 4)
        sample_num_train = sample_num + offset

        # indices = get_indices(batch_size, sample_num_train, point_num)
        indices = np.random.choice(P.size()[1], sample_num_train, replace = False).tolist()
        P_sampled = P[:,indices,:]
        F_sampled = F[:,indices,:]
        P_sampled = Variable(P_sampled).cuda()
        F_sampled = Variable(F_sampled).cuda()

        if False:
            P_draw = P_sampled.data.cpu().numpy()
            # fig = plt.figure()
            # ax = fig.gca(projection = '3d')
            # ax.scatter(P_draw[25,:,0], P_draw[25,:,1], P_draw[25,:,2], c = 'k')
            # plt.show()

            plt.style.use('grayscale')
            plt.axis([-3, 3, -3, 3])
            plt.scatter(P_draw[25,:,0], -P_draw[25,:,1], c = 1 - F_sampled[25,:,0], marker = ',', s = 25)
            print("LABEL:", label[25])
            plt.show()

        optimizer.zero_grad()

        t0 = time.time()
        out = model((P_sampled, F_sampled))

        loss = loss_fn(out, Variable(label.long()).cuda())
        loss.backward()
        optimizer.step()

        if global_step % 25 == 0:
            loss_v = loss.data[0]
            print("Loss:", loss_v)
        else:
            loss_v = 0

        if global_step % 250 == 0:

            # Testing accuracy
            accuracy_sum = 0
            testing_size = 4 # times testing_batch_size = 256
            for t, (data, label) in enumerate(testing_loader):
                if t >= testing_size:
                    break

                model = model.eval()

                P = data[:,:,:3]
                F = data[:,:,3:]

                offset = int(random.gauss(0, sample_num // 8))
                offset = max(offset, -sample_num // 4)
                offset = min(offset, sample_num // 4)
                sample_num_train = sample_num + offset

                # indices = get_indices(batch_size, sample_num_train, point_num)
                indices = np.random.choice(P.size()[1], sample_num_train, replace = False).tolist()
                P_sampled = P[:,indices,:]
                F_sampled = F[:,indices,:]
                P_sampled = Variable(P_sampled).cuda()
                F_sampled = Variable(F_sampled).cuda()

                out = model((P_sampled, F_sampled))
                probs = nn.Softmax()(out)
                _, pred = probs.max(1)
                accuracy_sum += torch.mean((pred.data.cpu() == label.long()).float())
            accuracy = accuracy_sum / testing_size
            print("accuracy:", accuracy)

            losses.append(loss_v)
            accuracies.append(accuracy)

        global_step += 1
