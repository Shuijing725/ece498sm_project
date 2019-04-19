import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io, transform
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision
from torchvision import transforms, utils
import torchvision.utils as vutils
import torchvision.datasets as dset
from torch.autograd import Variable
import torchvision.models as models


# Number of workers for dataloader
workers = 4

# Batch size during training
batch_size = 128

# Spatial size of training images. The image size of original images vary from 15 to 250. 
#But to train our CNN we must have the fixed size for the inputs.
#All images will be resized to this size using a transformer.
###### TO BE CHANGED!!!!!!!!!!!!!!!!!!!!!!!!
image_size = 500

# Number of training epochs
num_epochs = 15

#number of training classes
num_classes = 2

# Learning rate for optimizers
learning_rate = 0.002

# Number of GPUs available. Use 0 for CPU mode.
num_gpu = 1

# Set random seed for reproducibility
manualSeed = 999
torch.manual_seed(manualSeed)


# Load the data
# TO BE CHANGED!!!!!!!
trainingClassifierRoot = '/home/peixin/mp2_data/Final_Training_flip/Images'
testClassifierRoot = '/home/peixin/mp2_data/Online-Test-sort/'
####
normImgTensor = transforms.Normalize(mean=np.zeros(3), std=np.ones(3))

# Create the dataset
trainClassifierDataset = dset.ImageFolder(root=trainingClassifierRoot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							   normImgTensor
                           ]))

testClassifierDataset = dset.ImageFolder(root=testClassifierRoot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							   normImgTensor
                           ]))

# Create the dataloader
trainClassifierLoader = torch.utils.data.DataLoader(trainClassifierDataset, batch_size=batch_size, shuffle=True, num_workers=workers)
testClassifierLoader = torch.utils.data.DataLoader(testClassifierDataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# Decide which device(GPU or CPU) we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")


# ### Design the model
class TrafficSignClassifier(nn.Module):
    def __init__(self, num_gpu):
        super(TrafficSignClassifier, self).__init__()
        self.ngpu = num_gpu

       	self.fc_conv = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
		self.fc_conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)

		self.conv_ncl = nn.Conv2d(1024, num_classes, 1)
		self.up_conv1 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2)
        ####

    # the architecture is taken from Table I of https://ieeexplore.ieee.org/abstract/document/7759717
    def forward(self, x):
        x = models.vgg16(pretrained=True)
		# remove the fully-connected layers out
		x = x.features
		# add the transpose convolutional layers
		x = self.fc_conv(x)
		x = self.fc_conv2(x)
		x = self.conv_ncl(x)
		for i in range(5):
			x = self.up_conv1(x)

		return x


# In[8]:


### Choose different CNN architectures here! ###

classifier = TrafficSignClassifier(num_gpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (num_gpu > 0):
    classifier = nn.DataParallel(classifier, list(range(num_gpu)))
    
#To load the trained weights
#Uncomment the following line to load the weights. Then you can run testing directly or continue the training
#classfier.load_state_dict(torch.load('./MP2weights.pth', map_location='cpu')) 
#classfier.eval()

#Print the model
# print(classifier)


# ### Set up training environment
# You can choose different loss functions and optimizers. Here I just use the same ones as in PyTorch official tutorial.

# In[9]:


#training parameters
##TODO
criterion = nn.CrossEntropyLoss()#loss function
optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
####


# Now let's start the training. Just like in tutorial, you can print out your loss and the running time at the end of each epoch to monitor the training process. 

# In[10]:


def calculate_val_accuracy(valloader):
	correct = 0.
	total = 0.
	predictions = []

	class_correct = list(0. for i in range(num_classes))
	class_total = list(0. for i in range(num_classes))

	for data in valloader:
		images, labels = data
		if num_gpu > 0:
			images = images.cuda()
			labels = labels.cuda()
		outputs = classifier(Variable(images))
		_, predicted = torch.max(outputs.data, 1)
		predictions.extend(list(predicted.cpu().numpy()))
		total += labels.size(0)
		correct += (predicted == labels).sum()

		c = (predicted == labels).squeeze()
		for i in range(len(labels)):
			label = labels[i]
			class_correct[label] += c[i]
			class_total[label] += 1

	class_accuracy = 100.0 * np.divide(class_correct, class_total)
	return 100.0*correct/total, class_accuracy


# In[11]:



# Training Loop, no need to run if you already loaded the weights
for epoch in range(20):  # loop over the dataset multiple times  
	running_loss = 0.0
	for i, data in enumerate(trainClassifierLoader, 0):
		# get the inputs
		inputs, labels = data
		if num_gpu > 0:
			inputs = inputs.cuda()
			labels = labels.cuda()
	
		# zero out the parameter gradients
		#Every time a variable is back propogated through, the gradient will be accumulated instead of being replaced. 
		optimizer.zero_grad()
	
		# forward + backward + optimize
		outputs = classifier(inputs)
		loss = criterion(outputs, labels)
	
		#loss.backward() computes dloss/dx for every parameter x
		loss.backward()	
	
		#optimizer.step updates the value of x using the gradient x.grad.
		optimizer.step() 
	
		# print statistics
		running_loss += loss.item()
		
		# if i % 100 == 99:    # print every 2000 mini-batches
		# 	print('[%d, %5d] loss: %.3f' %
		# 		(epoch + 1, i + 1, running_loss / 2000))
		# 	running_loss = 0.0
	
	acc, class_acc = calculate_val_accuracy(trainClassifierLoader)
	print('epoch', epoch, ': loss = ', running_loss, ' accuracy = %d %%' % acc)

print('Finished Training')


# ### Test the Model
# Now our model training is finished. If you are satisfied by the result from part of the test set, let's try it on all testing images. Print out the accuracy on all test images. Note you need to achieve 97% accuracy to get full grade.

# In[12]:


test_acc, test_class_acc = calculate_val_accuracy(testClassifierLoader)
print('accuracy = %d %%' % test_acc)


# ### Analyze the Results for Each Individual Class
# To help futher improve the accuracy, we want to know for which classes our NN works well and for which classes it fails. Print out the accuracy of your classifier on each class on the whole testing dataset. Try to explain why some classess have low accuracy in your report. You don't have to speficy the name of each class. Just use something like "class 0" is good enough. 

# In[13]:


class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
classes = list(range(num_classes))
with torch.no_grad():
	for data in testClassifierLoader:
		images, labels = data
		images = images.cuda()
		labels = labels.cuda()
		outputs = classifier(images)
		_, predicted = torch.max(outputs, 1)
		c = (predicted == labels).squeeze()
		for i in range(labels.size(0)):
			label = labels[i]
			class_correct[label] += c[i].item()
			class_total[label] += 1


for i in range(num_classes):
	print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# ### Save Trained Weights
# Notice that the trained weights are just variables right now and will be lost when you close the Jupyter file. Obviously, you don't want to train the model again and again. PyTorch can help you save your work. Read the "Saving & Loading Model for Inference" part from this tutorial: https://pytorch.org/tutorials/beginner/saving_loading_models.html.<br>
# Please save the weights from your best model into "MP2weights.pth" and include it in your submission. TAs will not train the CNN for you. So if we cannot find this file, you will lose a lot of points.  

# In[12]:


#save the weights into "./MP2weights.pth"
torch.save(classifier.state_dict(), "./cnn_inception.pth")

