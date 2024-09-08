def q6_1_1():
	import numpy as np
	import scipy.io
	import torch
	import torch.nn as nn
	import matplotlib.pyplot as plt
	from torch.autograd import Variable
	import skimage.measure
	import torch.optim as optim
	import sys
	def get_random_batches(x, y, batch_size):
	    # Combine the data and labels
	    combined = list(zip(x, y))
	    
	    # Shuffle the combined data and labels
	    np.random.shuffle(combined)
	    
	    # Split the data back into separate lists
	    x[:], y[:] = zip(*combined)
	    
	    # Create batches
	    batches = []
	    for i in range(0, len(x), batch_size):
		x_batch = x[i:i + batch_size]
		y_batch = y[i:i + batch_size]
		
		# Convert batches to torch tensors and send to device
		x_batch = torch.from_numpy(np.array(x_batch)).float().to(device)
		y_batch = torch.from_numpy(np.array(y_batch)).long().to(device)
		
		batches.append((x_batch, y_batch))
	    
	    return batches


	# Check for GPU availability
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using device:", device)

	# Load your data
	train_data = scipy.io.loadmat('../data/nist36_train.mat')
	valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

	train_x, train_y = train_data['train_data'], train_data['train_labels']
	valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

	# Convert validation data to torch tensors and send to device
	valid_x = torch.from_numpy(valid_x).float().to(device)
	valid_y = torch.from_numpy(valid_y).long().to(device)

	max_iters = 150
	batch_size = 32
	learning_rate = 0.002
	hidden_size = 64

	# Assuming get_random_batches is a function you've defined elsewhere
	batches = get_random_batches(train_x, train_y, batch_size)
	batch_num = len(batches)

	# Define the model and move it to the device
	model = nn.Sequential(
	    nn.Linear(train_x.shape[1], hidden_size),
	    nn.Sigmoid(),
	    nn.Linear(hidden_size, hidden_size),
	    nn.Sigmoid(),
	    nn.Linear(hidden_size, train_y.shape[1]),
	).to(device)

	training_loss_data = []
	valid_loss_data = []
	training_acc_data = []
	valid_acc_data = []

	# Define the loss function and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

	for itr in range(max_iters):
	    total_loss = 0
	    total_acc = 0

	    for xb, yb in batches:
		# Convert batch to torch tensors and send to device
		label = np.where(yb.cpu().numpy() == 1)[1]  # Move tensor to cpu for numpy conversion
		label = torch.tensor(label).to(device)

		optimizer.zero_grad()
		out = model(xb)
		loss = criterion(out, label)
		loss.backward()
		optimizer.step()

		_, predicted = torch.max(out.data, 1)
		total_acc += ((label == predicted).sum().item())
		total_loss += loss.item()

	    ave_acc = total_acc / train_x.shape[0]
	    ave_loss = total_loss / batch_num

	    valid_label = torch.tensor(np.where(valid_y.cpu().numpy() == 1)[1], device=device)
	    valid_out = model(valid_x)
	    valid_loss = criterion(valid_out, valid_label)
	    _, valid_predicted = torch.max(valid_out.data, 1)
	    valid_acc = (valid_label == valid_predicted).sum().item() / valid_x.shape[0]

	    training_loss_data.append(ave_loss)
	    valid_loss_data.append(valid_loss.item())
	    training_acc_data.append(ave_acc)
	    valid_acc_data.append(valid_acc)

	    if itr % 2 == 0:
		print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, ave_loss, ave_acc))

	print('Validation accuracy: ', valid_acc)

	plt.figure(0)
	plt.xlabel('Number of iterations')
	plt.ylabel('Accuracy')
	plt.plot(np.arange(max_iters), training_acc_data, 'g')
	plt.plot(np.arange(max_iters), valid_acc_data, 'r')
	plt.legend(['training accuracy', 'validation accuracy'])
	plt.show()

	plt.figure(1)
	plt.xlabel('Number of iterations')
	plt.ylabel('Training loss')
	plt.plot(np.arange(max_iters), training_loss_data, 'g')
	plt.plot(np.arange(max_iters), valid_loss_data, 'r')
	plt.legend(['training loss', 'validation loss'])
	plt.show()

def q6_1_2():
	import numpy as np
	import scipy.io
	import torch
	import torchvision
	import torchvision.transforms as transforms
	import matplotlib.pyplot as plt
	from torch.autograd import Variable
	import skimage.measure
	import torch.optim as optim

	max_iters = 20

	batch_size = 100
	learning_rate = 1e-2
	hidden_size = 64

	# Check for CUDA
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# NIST36 dataset
	train_data = scipy.io.loadmat('../data/nist36_train.mat')
	valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

	train_x, train_y = train_data['train_data'], train_data['train_labels']
	valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
	train_examples = train_x.shape[0]

	test_examples = valid_x.shape[0]
	train_x = torch.tensor(train_x).float().to(device)
	label = np.where(train_y == 1)[1]
	label = torch.tensor(label).to(device)

	train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(train_x, label),
		                                   batch_size=batch_size,
		                                   shuffle=True)
	valid_x = torch.tensor(valid_x).float().to(device)
	valid_label = np.where(valid_y == 1)[1]
	valid_label = torch.tensor(valid_label).to(device)

	test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(valid_x, valid_label),
		                                  batch_size=batch_size,
		                                  shuffle=True)

	# Define the Neural Network
	class Net(torch.nn.Module):
	    def __init__(self):
		super(Net, self).__init__()
		self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
		self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
		self.fc1 = torch.nn.Linear(5 * 5 * 50, 512)
		self.fc2 = torch.nn.Linear(512, 36)

	    def forward(self, x):
		x = torch.nn.functional.relu(self.conv1(x))
		x = torch.nn.functional.max_pool2d(x, 2, 2)
		x = torch.nn.functional.relu(self.conv2(x))
		x = torch.nn.functional.max_pool2d(x, 2, 2)
		x = x.view(-1, 5* 5 * 50)
		x = torch.nn.functional.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	# Create the model and move it to the device (GPU or CPU)
	model = Net().to(device)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

	training_loss_data = []
	valid_loss_data = [2.5]
	training_acc_data = []
	valid_acc_data = []
	# Training Loop
	for itr in range(max_iters):
	    total_loss = 0
	    total_acc = 0
	    valid_total_loss = 0
	    valid_total_acc = 0

	    model.train()
	    for batch_idx, (x, target) in enumerate(train_loader):
		x, target = x.to(device), target.to(device)
		x = x.reshape(batch_size,1,32,32)
		out = model(x)
		loss = criterion(out, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		_, predicted = torch.max(out.data, 1)
		total_acc += ((target == predicted).sum().item())
		total_loss += loss.item()

	    ave_acc = total_acc / train_examples

	    model.eval()
	    with torch.no_grad():
		for batch_idx, (valid_x, valid_target) in enumerate(test_loader):
		    valid_x, valid_target = valid_x.to(device), valid_target.to(device)
		    valid_x = valid_x.reshape(batch_size, 1, 32, 32)
		    valid_out = model(valid_x)
		    valid_loss = criterion(valid_out, valid_target)
		    _, valid_predicted = torch.max(valid_out.data, 1)
		    valid_total_acc += ((valid_target == valid_predicted).sum().item())
		    valid_total_loss += valid_loss.item()

	    valid_acc = valid_total_acc / test_examples

	    training_loss_data.append(total_loss / (train_examples / batch_size))
	    valid_loss_data.append((valid_loss / (test_examples / batch_size)).cpu())
	    training_acc_data.append(ave_acc)
	    valid_acc_data.append(valid_acc)
	    print('Validation accuracy: ', valid_acc)
	    if itr % 2 == 0:
		print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, ave_acc))

	valid_loss_data.pop()
	# Plotting
	plt.figure(0)
	plt.xlabel('Number of Iterations')
	plt.ylabel('Accuracy')
	plt.plot(np.arange(max_iters), training_acc_data, 'g')
	plt.plot(np.arange(max_iters), valid_acc_data, 'r')
	plt.legend(['Training Accuracy', 'Validation Accuracy'])
	plt.show()

	plt.figure(1)
	plt.xlabel('Number of Iterations')
	plt.ylabel('Loss')
	plt.plot(np.arange(max_iters), training_loss_data, 'g')
	plt.plot(np.arange(max_iters), valid_loss_data, 'r')
	plt.legend(['Training Loss', 'Validation Loss'])
	plt.show()

def q6_1_3():
	import torch
	import torch.nn as nn
	import torch.optim as optim
	import torchvision
	import torchvision.transforms as transforms
	import torchvision.models as models
	import matplotlib.pyplot as plt

	# Check if CUDA is available, else use CPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load the pre-trained MobileNetV2 model and modify it for CIFAR-10
	model = models.mobilenet_v2(pretrained=True)
	# CIFAR-10 has 10 classes, so we need to modify the last layer of the model
	model.classifier[1] = nn.Linear(model.last_channel, 10)
	model.to(device)

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	# Transformations for the input data
	transform = transforms.Compose([
	    transforms.Resize(224),  # Resize images to the input size of MobileNetV2
	    transforms.ToTensor(),
	    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	# Loading CIFAR10 dataset
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

	# Training the model
	num_epochs = 10
	train_losses = []
	train_accuracies = []

	for epoch in range(num_epochs):
	    running_loss = 0.0
	    correct = 0
	    total = 0

	    for i, data in enumerate(trainloader, 0):
		inputs, labels = data[0].to(device), data[1].to(device)

		# Forward pass
		outputs = model(inputs)
		loss = criterion(outputs, labels)

		# Backward pass and optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Calculate accuracy
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

		running_loss += loss.item()

	    epoch_loss = running_loss / len(trainloader)
	    epoch_accuracy = 100 * correct / total
	    train_losses.append(epoch_loss)
	    train_accuracies.append(epoch_accuracy)
	    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

	# Plotting training accuracy and loss
	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(range(num_epochs), train_losses, label='Training Loss')
	plt.title('Training Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(range(num_epochs), train_accuracies, label='Training Accuracy')
	plt.title('Training Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()

	plt.show()


	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	testloader = DataLoader(testset, batch_size=4, shuffle=False)

	# Evaluate the model on test data
	model.eval()  # Set the model to evaluation mode
	test_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
	    for data in testloader:
		images, labels = data[0].to(device), data[1].to(device)
		outputs = model(images)
		loss = criterion(outputs, labels)
		test_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	test_loss /= len(testloader)
	test_accuracy = 100 * correct / total
	print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

def q6_1_4():
	import torch
	import torch.nn as nn
	import torch.optim as optim
	import torchvision.models as models
	from torch.utils.data import Dataset, DataLoader
	from torchvision import transforms
	from PIL import Image
	import pandas as pd
	import os

	# Custom Dataset class
	class SUNDataSet(Dataset):
	    def __init__(self, files_path, labels_path, root_dir, transform=None):
		self.file_names = pd.read_csv(files_path, header=None).values.squeeze()
		self.labels = pd.read_csv(labels_path, header=None).values.squeeze()
		self.root_dir = root_dir
		self.transform = transform

	    def __len__(self):
		return len(self.file_names)

	    def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.file_names[idx])
		image = Image.open(img_name)
		label = self.labels[idx]
		if self.transform:
		    image = self.transform(image)
		return image, label

	# Check if CUDA (GPU support) is available and set the device accordingly
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Define transformations
	transform = transforms.Compose([
	    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
	    transforms.Resize((224, 224)),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	# Root directory for the dataset
	root_dir = '../data/data/'

	# Load the datasets
	train_dataset = SUNDataSet(os.path.join(root_dir, 'train_files.txt'), 
		                   os.path.join(root_dir, 'train_labels.txt'), 
		                   root_dir, 
		                   transform=transform)
	test_dataset = SUNDataSet(os.path.join(root_dir, 'test_files.txt'), 
		                  os.path.join(root_dir, 'test_labels.txt'), 
		                  root_dir, 
		                  transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

	# Load pre-trained MobileNetV2 and modify it
	model = models.mobilenet_v2(pretrained=True)
	num_classes = len(set(train_dataset.labels))  # Assuming this gives the correct number of classes
	model.classifier[1] = nn.Linear(model.last_channel, num_classes)
	model.to(device)

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	# Training loop
	num_epochs = 10
	train_losses = []
	train_accuracies = []

	for epoch in range(num_epochs):
	    model.train()
	    running_loss = 0.0
	    correct = 0
	    total = 0

	    for i, (inputs, labels) in enumerate(train_loader):
		inputs, labels = inputs.to(device), labels.to(device)

		# Forward pass
		outputs = model(inputs)
		loss = criterion(outputs, labels)

		# Backward pass and optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Calculate accuracy
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

		running_loss += loss.item()

	    epoch_loss = running_loss / len(train_loader)
	    epoch_accuracy = 100 * correct / total
	    train_losses.append(epoch_loss)
	    train_accuracies.append(epoch_accuracy)
	    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

	# Evaluation loop
	model.eval()
	test_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
	    for inputs, labels in test_loader:
		inputs, labels = inputs.to(device), labels.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		test_loss += loss.item()

	test_loss = test_loss / len(test_loader)
	test_accuracy = 100 * correct / total
	print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

	import matplotlib.pyplot as plt

	# Assuming you have stored training losses and accuracies in 'train_losses' and 'train_accuracies' respectively
	# and they are available as lists

	# Plotting the training loss
	plt.figure(figsize=(12, 6))

	plt.subplot(1, 2, 1)
	plt.plot(train_losses, label='Train Loss')
	plt.title('Training Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()

	# Plotting the training accuracy
	plt.subplot(1, 2, 2)
	plt.plot(train_accuracies, label='Train Accuracy')
	plt.title('Training Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()

	plt.show()

def q6_2_1():
	import torch
	import torch.nn as nn
	import torch.optim as optim
	import torchvision
	import torchvision.models as models
	import torchvision.transforms as transforms
	from torch.utils.data import DataLoader, Dataset
	from torchvision.datasets import ImageFolder
	import os

	# Custom CNN Architecture
	class CustomCNN(nn.Module):
	    def __init__(self, num_classes):
		super(CustomCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(128 * 28 * 28, 512)
		self.fc2 = nn.Linear(512, num_classes)
		self.dropout = nn.Dropout(0.5)

	    def forward(self, x):
		x = self.pool(torch.relu(self.conv1(x)))
		x = self.pool(torch.relu(self.conv2(x)))
		x = self.pool(torch.relu(self.conv3(x)))
		x = x.view(-1, 128 * 28 * 28)
		x = torch.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		return x

	# Check if CUDA is available and set the device accordingly
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Define transformations for the input data
	transform = transforms.Compose([
	    transforms.Resize((224, 224)),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	# Load the datasets
	train_data = ImageFolder(root='../data/oxford-flowers17/train', transform=transform)
	val_data = ImageFolder(root='../data/oxford-flowers17/val', transform=transform)
	test_data = ImageFolder(root='../data/oxford-flowers17/test', transform=transform)

	train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
	test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

	# Load pre-trained SqueezeNet and modify it for fine-tuning
	squeezenet = models.squeezenet1_1(pretrained=True)
	num_classes = len(train_data.classes)
	squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
	squeezenet.num_classes = num_classes
	squeezenet.to(device)

	# Initialize the custom CNN
	custom_cnn = CustomCNN(num_classes)
	custom_cnn.to(device)

	# Define the loss function and optimizer for both models
	criterion = nn.CrossEntropyLoss()
	optimizer_sqz = optim.Adam(squeezenet.parameters(), lr=0.001)
	optimizer_custom = optim.Adam(custom_cnn.parameters(), lr=0.001)

	# Training loop for both models
	def train_model(model, optimizer, num_epochs=100):
	    model.train()
	    losses = []
	    accuracies = []
	    for epoch in range(num_epochs):
		running_loss = 0.0
		correct = 0
		total = 0

		for i, (inputs, labels) in enumerate(train_loader):
		    inputs, labels = inputs.to(device), labels.to(device)

		    # Forward pass
		    outputs = model(inputs)
		    loss = criterion(outputs, labels)

		    # Backward pass and optimization
		    optimizer.zero_grad()
		    loss.backward()
		    optimizer.step()

		    # Calculate accuracy
		    _, predicted = torch.max(outputs.data, 1)
		    total += labels.size(0)
		    correct += (predicted == labels).sum().item()

		    running_loss += loss.item()
		    

		epoch_loss = running_loss / len(train_loader)
		epoch_accuracy = 100 * correct / total
		print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
		losses.append(epoch_loss)
		accuracies.append(epoch_accuracy)
		
	    return losses, accuracies
		

	# Train SqueezeNet
	print("Training SqueezeNet")
	train_losses_sqz, train_accuracies_sqz = train_model(squeezenet, optimizer_sqz, num_epochs=100)

	# Train Custom CNN
	print("\nTraining Custom CNN")
	train_losses_custom, train_accuracies_custom = train_model(custom_cnn, optimizer_custom)

	import matplotlib.pyplot as plt

	# Plotting
	plt.figure(figsize=(12, 8))

	# Training Loss
	plt.subplot(2, 2, 1)
	plt.plot(train_losses_sqz, label='SqueezeNet')
	plt.plot(train_losses_custom, label='Custom CNN')
	plt.title('Training Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()

	# Training Accuracy
	plt.subplot(2, 2, 2)
	plt.plot(train_accuracies_sqz, label='SqueezeNet')
	plt.plot(train_accuracies_custom, label='Custom CNN')
	plt.title('Training Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend()


	plt.tight_layout()
	plt.show()

	# Define a function for model evaluation
	def evaluate_model(model, test_loader):
	    model.eval()  # Set the model to evaluation mode
	    correct = 0
	    total = 0
	    with torch.no_grad():
		for inputs, labels in test_loader:
		    inputs, labels = inputs.to(device), labels.to(device)
		    outputs = model(inputs)
		    _, predicted = torch.max(outputs.data, 1)
		    total += labels.size(0)
		    correct += (predicted == labels).sum().item()

	    accuracy = 100 * correct / total
	    return accuracy

	# Evaluate SqueezeNet
	accuracy_sqz = evaluate_model(squeezenet, test_loader)
	print(f"SqueezeNet Test Accuracy: {accuracy_sqz}%")

	# Evaluate Custom CNN
	accuracy_custom = evaluate_model(custom_cnn, test_loader)
	print(f"Custom CNN Test Accuracy: {accuracy_custom}%")

	# Compare the models
	print("\nComparison:")
	print(f"SqueezeNet Accuracy: {accuracy_sqz}%")
	print(f"Custom CNN Accuracy: {accuracy_custom}%")
