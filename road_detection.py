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
from distutils.version import LooseVersion


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
				low_red = (240, 0, 240)
				high_red = (255, 0, 255)

				# masked_img: 2d array, red: 255, else: 0
				img = cv2.inRange(img, low_red, high_red)
				plt.imshow(img)
				img = cv2.resize(img, (output_size, output_size))
				# normalize img to [0, 1]
				_, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

				# change img to channel first (w, h, 3) -> (3, w, h)
				# img = np.moveaxis(img, -1, 0)  # (3, 224, 224)
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
		self.up_conv2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2)
		self.up_conv3 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2)
		self.up_conv4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2)
		self.up_conv5 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2)

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
		x = self.up_conv2(x)
		x = self.relu(x)

		x = self.up_conv3(x)
		x = self.relu(x)
		x = self.up_conv4(x)
		x = self.relu(x)
		x = self.up_conv5(x)
		x = self.relu(x)

		return x


# def calculate_val_accuracy(valloader):
# 	correct = 0.
# 	total = 0.
# 	predictions = []
#
# 	class_correct = list(0. for i in range(num_classes))
# 	class_total = list(0. for i in range(num_classes))
#
# 	for data in valloader:
# 		images, labels = data
# 		if num_gpu > 0:
# 			images = images.cuda()
# 			labels = labels.cuda()
# 		outputs = classifier(Variable(images))
# 		_, predicted = torch.max(outputs.data, 1)
# 		predictions.extend(list(predicted.cpu().numpy()))
# 		total += labels.size(0)
# 		correct += (predicted == labels).sum()
#
# 		c = (predicted == labels).squeeze()
# 		for i in range(len(labels)):
# 			label = labels[i]
# 			class_correct[label] += c[i]
# 			class_total[label] += 1
#
# 	class_accuracy = 100.0 * np.divide(class_correct, class_total)
# 	return 100.0*correct/total, class_accuracy


def cross_entropy2d(input, target, weight=None, size_average=True):
	# input: (n, c, h, w), target: (n, h, w)
	n, c, h, w = input.size()
	# log_p: (n, c, h, w)
	if LooseVersion(torch.__version__) < LooseVersion('0.3'):
		# ==0.2.X
		log_p = F.log_softmax(input)
	else:
		# >=0.3
		log_p = F.log_softmax(input, dim=1)
	# log_p: (n*h*w, c)
	log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
	log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
	log_p = log_p.view(-1, c)
	# target: (n*h*w,)
	mask = target >= 0
	target = target[mask]
	target = target.type(torch.LongTensor).cuda()
	loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
	if size_average:
		loss /= mask.data.sum()
	return loss

def train():
	# Batch size during training
	batch_size = 20

	# Spatial size of training images. The image size of original images vary from 15 to 250.
	# But to train our CNN we must have the fixed size for the inputs.
	# All images will be resized to this size using a transformer.
	img_size = 224
	output_size = 222
	# Number of training epochs
	num_epochs = 80

	# number of training classes
	num_classes = 1

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
	train_data, _ = load_images(img_path, batch_size, img_size, train_test='train', device = device)
	train_label, _ = load_ground_truth(label_path, batch_size, output_size, road_lane='road', device = device)


	classifier = RoadDetector(num_gpu).to(device)
	# training parameters
	criterion = nn.MSELoss()  # loss function
	# optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
	optimizer = optim.Adam(classifier.parameters())
	# Training Loop, no need to run if you already loaded the weights
	for epoch in range(num_epochs):  # loop over the dataset multiple times
		print("epoch:", epoch)
		running_loss = 0.0
		for inputs, labels in zip(train_data, train_label):
			inputs = Variable(inputs, requires_grad=True)
			labels = Variable(labels, requires_grad = True)
			# print("inputs:", inputs.size())
			# print("labels:", labels.size())
			# zero out the parameter gradients
			# Every time a variable is back propogated through, the gradient will be accumulated instead of being replaced.
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = classifier.forward(inputs)
			# print("outputs:", outputs.size())
			# labels = labels.type(torch.int64).cpu()
			# outputs = outputs.type(torch.int64).cpu()
			# loss = cross_entropy2d(outputs, labels)
			loss = criterion(outputs, labels)
			# print("loss:", loss)
			# print("#" * 30)
			# print(loss.grad, loss.is_leaf)
			# print("#" * 30)
			# loss.backward() computes dloss/dx for every parameter x
			loss.backward()

			# optimizer.step updates the value of x using the gradient x.grad.
			optimizer.step()


			# print statistics
			running_loss += loss
		print('loss = ', running_loss)

	print('Finished Training')
	torch.save(classifier.state_dict(), "./fcn.pth")

def test(model_path, batch_size, img_size, output_size, num_gpu = 1):
	device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

	test_img_path = '/home/shuijing/Desktop/ece498sm_project/data_road/testing/image_2'
	test_data, test_data_labels = load_images(test_img_path, batch_size, img_size, train_test='test', device=device)
	pred_path = '/home/shuijing/Desktop/ece498sm_project/predictions'

	model = RoadDetector(num_gpu).to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()
	if not os.path.exists(pred_path):
		os.makedirs(pred_path)
	# testing
	for img_batch, label_batch in zip(test_data, test_data_labels):
		for img, label in zip(img_batch, label_batch):
			img = torch.unsqueeze(img, 0)
			pred = model.forward(Variable(img, requires_grad=True))
			# print('pred:', pred)
			pred = (Variable(pred).data).cpu().numpy()  # convert tensor to numpy
			img = pred[0] # (1, 1, 222, 222)
			# print(pred.shape)
			# idx = np.argmax(img, axis = 0)
			# white = idx == 0
			# black = idx == 1
			# save_img = np.zeros((222, 222))
			# save_img[white] = 255
			# save_img[black] = 0
			img[img > 0.5] = 255
			img[img <= 0.5] = 0
			img = np.moveaxis(img, 0, -1)
			# print(pred.shape)
			# pred = Image.fromarray(pred, 'L')
			save_img = cv2.resize(img, output_size)

			# print(pred.shape)
			cv2.imwrite(os.path.join(pred_path, str(label + '.png')), save_img)


train()
test("./fcn.pth", 20, 224, (1242, 375))