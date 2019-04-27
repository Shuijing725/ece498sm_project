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


# load the original images from folder
# return value: nested list of images, seperated by batch_size
def load_images(path, batch_size, img_size, train_test='train', num_gpu = 1, device = None):
	# sanity check
	if num_gpu > 0:
		assert (device is not None)

	labels = []
	label_batch = []
	img_batches = None
	cur_batch = None
	count = 0
	if train_test == 'train':
		l = [('um', 95), ('umm', 96), ('uu', 98)]
	else:
		l = [('um', 96), ('umm', 94), ('uu', 100)]
	# load the 3 sets of images from path
	for prefix, tot in l:
		for i in range(tot):
			# obtain the full path of each image
			if i < 10:
				img_path = os.path.join(path, str(prefix + '_00000' + str(i)+'.png'))
			else:
				img_path = os.path.join(path, str(prefix + '_0000' + str(i)+'.png'))
			img = cv2.imread(img_path)
			img = cv2.resize(img, (img_size, img_size)) # (224, 224, 3)
			# normalize img to [0, 1]
			img  = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			# change img to channel first (w, h, 3) -> (3, w, h)
			img = np.moveaxis(img, -1, 0) # (3, 224, 224)
			# convert img to torch tensor
			img = torch.from_numpy(img)
			# convert the type of img to be float tensor
			img = img.type(torch.FloatTensor)
			if num_gpu > 0:
				img = img.to(device)
			img = torch.unsqueeze(img, 0) # (3, 224, 224) -> (1, 3, 224, 224)
			# append image label
			label_batch.append(str(prefix + '_road_0000' + str(i)))
			if count % batch_size == 0:
				# if cur_batch is full with batch_size images
				if cur_batch is not None:
					# append label_batch
					labels.append(label_batch)
					label_batch = []
					# append cur_batch to img_batches
					if img_batches is not None:
						cur_batch = torch.unsqueeze(cur_batch, 0)
						img_batches = torch.cat((img_batches, cur_batch), 0)
					else:
						img_batches = torch.unsqueeze(cur_batch, 0)
				# append this img to cur_batch
				cur_batch = img
			else:
				cur_batch = torch.cat((cur_batch, img), 0)
			count += 1

	return img_batches, labels

# load the ground truth lane markings from folder and convert them to the same form as the output of CNN
# return value: masked 2D image, with red = 255, pink = 0
def load_ground_truth(path, batch_size, output_size, road_lane = 'road', num_gpu = 1, device = None):
	# sanity check
	if num_gpu > 0:
		assert (device is not None)

	img_batches = None
	cur_batch = None
	count = 0
	labels = []
	label_batch = []
	if road_lane == 'road':
		# load the 3 sets of images from path
		for prefix, tot in [('um', 95), ('umm', 96), ('uu', 98)]:
			for i in range(tot):
				# obtain the full path of each image
				if i < 10:
					img_path = os.path.join(path, str(prefix + '_road_00000' + str(i) + '.png'))
				else:
					img_path = os.path.join(path, str(prefix + '_road_0000' + str(i) + '.png'))
				img = cv2.imread(img_path)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				low_red = (240, 0, 0)
				high_red = (255, 10, 10)
				# masked_img: 2d array, red: 255, else: 0
				img = cv2.inRange(img, low_red, high_red)
				img = cv2.resize(img, (output_size, output_size))
				# normalize img to [0, 1]
				img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
				# change img to channel first (w, h, 3) -> (3, w, h)
				img = np.moveaxis(img, -1, 0)  # (3, 224, 224)
				# convert img to torch tensor
				img = torch.from_numpy(img)
				# convert the type of img to be float tensor
				img = img.type(torch.FloatTensor)
				if num_gpu > 0:
					img = img.to(device)
				img = torch.unsqueeze(img, 0)  # (3, 224, 224) -> (1, 3, 224, 224)
				if num_gpu > 0:
					img = img.to(device)
				img = torch.unsqueeze(img, 0)  # (3, 224, 224) -> (1, 3, 224, 224)
				# append image label
				label_batch.append(str(prefix + '_road_0000' + str(i)))
				if count % batch_size == 0:
					# if cur_batch is full with batch_size images
					if cur_batch is not None:
						# append label_batch
						labels.append(label_batch)
						label_batch = []
						# append cur_batch to img_batches
						if img_batches is not None:
							cur_batch = torch.unsqueeze(cur_batch, 0)
							img_batches = torch.cat((img_batches, cur_batch), 0)
						else:
							img_batches = torch.unsqueeze(cur_batch, 0)
					# append this img to cur_batch
					cur_batch = img
				else:
					cur_batch = torch.cat((cur_batch, img), 0)
				count += 1


	else:
		for i in range(95):
			# obtain the full path of each image
			if i < 10:
				img_path = os.path.join(path, str('um_lane_00000' + str(i) + '.png'))
			else:
				img_path = os.path.join(path, str('um_lane_0000' + str(i) + '.png'))
			img = cv2.imread(img_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			low_red = (240, 0, 0)
			high_red = (255, 10, 10)
			# masked_img: 2d array, red: 255, else: 0
			img = cv2.inRange(img, low_red, high_red)
			img = cv2.resize(img, (output_size, output_size))
			# change img to channel first (w, h, 3) -> (3, w, h)
			img = np.moveaxis(img, -1, 0)  # (3, 224, 224)
			# convert img to torch tensor
			img = torch.from_numpy(img)
			# convert the type of img to be float tensor
			img = img.type(torch.FloatTensor)
			if num_gpu > 0:
				img = img.to(device)
			img = torch.unsqueeze(img, 0)  # (3, 224, 224) -> (1, 3, 224, 224)
			if num_gpu > 0:
				img = img.to(device)
			img = torch.unsqueeze(img, 0)  # (3, 224, 224) -> (1, 3, 224, 224)
			# append image label
			label_batch.append(str('um_lane_00000' + str(i)))
			if count % batch_size == 0:
				# if cur_batch is full with batch_size images
				if cur_batch is not None:
					# append label_batch
					labels.append(label_batch)
					label_batch = []
					# append cur_batch to img_batches
					if img_batches is not None:
						img_batches = torch.cat(img_batches, cur_batch)
					else:
						img_batches = torch.unsqueeze(cur_batch, 0)
				# append this img to cur_batch
				cur_batch = img
			else:
				cur_batch = torch.cat(cur_batch, img)
			count += 1

	return img_batches, labels


# ### Design the model
class RoadDetector(nn.Module):
	def __init__(self, num_gpu=1, num_classes=1):
		super(RoadDetector, self).__init__()
		self.ngpu = num_gpu

		self.fc_conv = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
		self.fc_conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)

		self.conv_ncl = nn.Conv2d(1024, num_classes, 1)
		self.up_conv1 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2)

		self.bn = nn.BatchNorm2d(1024)
		self.relu = nn.ReLU(inplace=True)


	# the architecture is taken from Table I of https://ieeexplore.ieee.org/abstract/document/7759717
	def forward(self, input):
		vgg = models.vgg16(pretrained=True).cuda()
		# remove the fully-connected layers out
		encoder = vgg.features
		x = encoder(input)
		# add the transpose convolutional layers
		x = self.fc_conv(x)
		x = self.relu(x)
		x = self.fc_conv2(x)
		x = self.bn(x)
		x = self.relu(x)
		x = self.conv_ncl(x)
		x = self.relu(x)

		# decoder
		x = self.up_conv1(x)
		x = self.relu(x)
		x = self.up_conv1(x)
		x = self.relu(x)

		x = self.up_conv1(x)
		x = self.relu(x)
		x = self.up_conv1(x)
		x = self.relu(x)
		x = self.up_conv1(x)
		x = self.relu(x)

		return x


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


# ### Analyze the Results for Each Individual Class
# To help futher improve the accuracy, we want to know for which classes our NN works well and for which classes it fails. Print out the accuracy of your classifier on each class on the whole testing dataset. Try to explain why some classess have low accuracy in your report. You don't have to speficy the name of each class. Just use something like "class 0" is good enough. 

#
# class_correct = list(0. for i in range(num_classes))
# class_total = list(0. for i in range(num_classes))
# classes = list(range(num_classes))
# with torch.no_grad():
# 	for data in testClassifierLoader:
# 		images, labels = data
# 		images = images.cuda()
# 		labels = labels.cuda()
# 		outputs = classifier(images)
# 		_, predicted = torch.max(outputs, 1)
# 		c = (predicted == labels).squeeze()
# 		for i in range(labels.size(0)):
# 			label = labels[i]
# 			class_correct[label] += c[i].item()
# 			class_total[label] += 1
#
#
# for i in range(num_classes):
# 	print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))

def main():
	# Batch size during training
	batch_size = 50

	# Spatial size of training images. The image size of original images vary from 15 to 250.
	# But to train our CNN we must have the fixed size for the inputs.
	# All images will be resized to this size using a transformer.
	###### TO BE CHANGED!!!!!!!!!!!!!!!!!!!!!!!!
	img_size = 224
	output_size = 222
	# Number of training epochs
	num_epochs = 15

	# number of training classes
	num_classes = 1

	# Learning rate for optimizers
	learning_rate = 0.002

	# Number of GPUs available. Use 0 for CPU mode.
	num_gpu = 1

	# Set random seed for reproducibility
	manualSeed = 999
	torch.manual_seed(manualSeed)

	# Decide which device(GPU or CPU) we want to run on
	device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

	# Load the data
	img_path = '/home/shuijing/Desktop/ece498sm_project/data_road/training/image_2'
	label_path = '/home/shuijing/Desktop/ece498sm_project/data_road/training/gt_image_2'
	test_img_path = '/home/shuijing/Desktop/ece498sm_project/data_road/testing/image_2'
	train_data, _ = load_images(img_path, batch_size, img_size, train_test='train', device = device)
	train_label, _ = load_ground_truth(label_path, batch_size, output_size, road_lane='road', device = device)
	test_data, test_data_labels = load_images(test_img_path, batch_size, img_size, train_test='test', device = device)


	classifier = RoadDetector(num_gpu).to(device)
	# training parameters
	criterion = nn.MSELoss()  # loss function
	optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

	# Training Loop, no need to run if you already loaded the weights
	for epoch in range(num_epochs):  # loop over the dataset multiple times
		print("epoch:", epoch)
		running_loss = 0.0
		for inputs, labels in zip(train_data, train_label):
			# print("inputs:", inputs.size())
			# print("labels:", labels.size())
			# zero out the parameter gradients
			# Every time a variable is back propogated through, the gradient will be accumulated instead of being replaced.
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = classifier(inputs)
			# print("outputs:", outputs.size())
			# labels = labels.type(torch.long)
			# outputs = outputs.type(torch.long)
			loss = criterion(outputs, labels)

			# loss.backward() computes dloss/dx for every parameter x
			loss.backward()

			# optimizer.step updates the value of x using the gradient x.grad.
			optimizer.step()

			# print statistics
			running_loss += loss.item()
		print('loss = ', running_loss)


	print('Finished Training')
	torch.save(classifier.state_dict(), "./fcn.pth")
	pred_path = '/home/shuijing/Desktop/ece498sm_project/predictions'
	if not os.path.exists(pred_path):
		os.makedirs(pred_path)
	# testing
	for img_batch, label_batch in zip(test_data, test_data_labels):
		for img, label in zip(img_batch, label_batch):
			img = torch.unsqueeze(img, 0)
			pred = classifier(img)
			print('pred:', pred)
			pred = (Variable(pred).data).cpu().numpy() # convert tensor to numpy
			print('pred1:', pred)
			pred = np.moveaxis(pred, 0, -1) # convert from (channel, row, col) to (row, col, channel)
			pred = cv2.resize(pred, (img_size, img_size))
			cv2.imwrite(os.path.join(pred_path, str(label + '.png')), pred)

	# save the weights into "./MP2weights.pth"
	# torch.save(classifier.state_dict(), "./cnn_inception.pth")

main()