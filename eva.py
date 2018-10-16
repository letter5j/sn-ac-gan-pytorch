from __future__ import print_function
import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import sn_m

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt



import itertools
from sklearn.metrics import confusion_matrix, classification_report
import os

from get_data import get_data_loaders

def demo2(s):
    mu ,sigma = 0, 1
    sampleNo = 1000
    # s = np.random.normal(mu, sigma, sampleNo)

    plt.hist(s, bins=500, normed=True)
    plt.savefig('true.png')
    plt.close()

def demo3():
    mu, sigma , num_bins = 0, 1, 50
    x = mu + sigma * np.random.randn(1000000)
    # 正态分布的数据
    n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor = 'blue', alpha = 0.5)
    # 拟合曲线
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Expectation')
    plt.ylabel('Probability')
    plt.title('histogram of normal distribution: $\mu = 0$, $\sigma=1$')

    plt.subplots_adjust(left = 0.15)
    plt.savefig('demo.png')
    plt.close()

# demo3()
# random.seed(5000)


nz = int(100)
ngf = int(64)
ndf = int(64)
nc = 3
num_classes = 7

netG = sn_m.netG(nz, ngf, nc)
netD = sn_m.netD(ndf, nc, num_classes)

netG.load_state_dict(torch.load('netG_epoch_49999.pth'))
netD.load_state_dict(torch.load('netD_epoch_49999.pth'))
netD.train(False)

netG.train(False)
# netD = sn_m.netD(ndf, nc, num_classes)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)


# generator images
class_name = ['Cellulitis', 'bruises', 'cut', 'mouth', 'sature', 'scrape', 'ulcer']


# for classNum in range(num_classes):


#     input = torch.FloatTensor(500, 3, 500, 500)
#     noise = torch.FloatTensor(500, nz, 1, 1)
#     fixed_noise = torch.FloatTensor(500, nz, 1, 1).normal_(0, 1)

#     dis_label = torch.FloatTensor(500)
#     aux_label = torch.LongTensor(500)

#     real_label = 1
#     fake_label = 0

#     netD.cuda()
#     netG.cuda()
#     input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
#     noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

#     input = Variable(input)
#     dis_label = Variable(dis_label)
#     aux_label = Variable(aux_label)
#     noise = Variable(noise)
#     fixed_noise = Variable(fixed_noise)
#     # fixed_noise_ = np.random.normal(0, 1, (500, nz))
#     fixed_noise_ = np.random.normal(0, 1, (500, nz))

#     # demo2(fixed_noise_)

#     # random_label = np.random.randint(0, num_classes, 500)
#     # random_label = np.zeros((500), dtype=int)
#     random_label = np.array((500), dtype=int)
#     random_label.fill(classNum) 

#     random_onehot = np.zeros((500, num_classes))

#     random_onehot[np.arange(500), random_label] = 1

#     fixed_noise_[np.arange(500), :num_classes] = random_onehot[np.arange(500)]

#     fixed_noise_ = (torch.from_numpy(fixed_noise_))
#     fixed_noise_ = fixed_noise_.resize_(500, nz, 1, 1)
#     fixed_noise.data.copy_(fixed_noise_)


#     fake = netG(fixed_noise)
#     print('fake:{}'.format(fake.shape))
#     for index, img in enumerate(fake.data):
#         vutils.save_image(img, '%s/%s_fake_samples_%s.png' %(class_name[classNum], class_name[classNum], str(index)))


# for i in range(100):
#     if (i >= num_classes):
#         for offset in [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:

#             unique_tensor = fixed_noise_.resize_(500, nz).numpy().copy()
#             unique_tensor[np.arange(500), i] = unique_tensor[np.arange(500), i] + offset
            
#             unique_tensor = (torch.from_numpy(unique_tensor))
#             unique_tensor = unique_tensor.resize_(500, nz, 1, 1)
#             fixed_noise.data.copy_(unique_tensor)

#             fake = netG(fixed_noise)
#             print('fake:{}'.format(fake.shape))
#             vutils.save_image(fake.data,
#                         'result/%s/fake_samples_%s.png' %(str(i), str(offset)))


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues,
#                           filename='confusion_matrix.png'):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)
#     plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     PATH = os.path.abspath(os.path.dirname(__file__))
#     plt.savefig(os.path.join(PATH, 'result', filename))
#     plt.close()


# def create_confusion_matrix(y_true, y_pred, class_names, model_name):



#     cnf_matrix = confusion_matrix(y_true, y_pred)
#     plot_confusion_matrix(cnf_matrix, classes=class_names,
#                           title='Confusion matrix', filename=('%s_confusion_matrix.png' %(model_name)))


# def cal_map(map_array):

#     result = []

#     for classification in map_array:
#         result.append(sum(classification)/len(classification))
#     return result

# def eval_model(model):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     ################## generate result
#     model.to(device)
#     dataloaders, dataset_sizes, class_names = get_data_loaders()

#     map_array = []
#     for i in range(len(class_names)):
#         map_array.append([])

#     y_pred = []
#     y_true = []

#         # Iterate over data.
#     for inputs, labels in dataloaders['test']:
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         # forward
#         dis_output, aux_output = model(inputs)
#         # m = torch.nn.Softmax(0)
#         # XD = m(outputs)
#         # print(outputs)
#         # print(XD)
#         _, preds = torch.max(aux_output.data, 1)
#         _, top_label = torch.topk(aux_output.data, len(class_names))

        
#         # print('predict ', preds)
#         # print('True ', labels.data)
#         y_pred += preds.cpu().numpy().flatten().tolist()
#         y_true += labels.data.cpu().numpy().flatten().tolist()

#         true_label = labels.data.cpu().numpy().flatten().tolist()
#         top_label = top_label.cpu().numpy().tolist()
#         # For mAP

#         for i in range(len(top_label)):
#             trueValue = true_label[i]
#             probability = 0.0
#             for j in range(len(top_label[i])):
#                 if(top_label[i][j] == trueValue):
#                     probability = 1 / (j + 1)
#             map_array[trueValue].append(probability)
        
#         # print(mapk(labels.data.cpu().numpy().flatten().tolist(), preds.cpu().numpy().flatten().tolist(), k=10))
    
#     create_confusion_matrix(y_true, y_pred, class_names, "netD")
#     print(classification_report(y_true, y_pred, target_names=class_names))
#     print(cal_map(map_array))
    
# eval_model(netD)
