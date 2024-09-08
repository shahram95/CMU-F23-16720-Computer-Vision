import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.01
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####import scipy.io
import numpy as np
##########################
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, train_y.shape[1], params, 'output')


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 )

for i in range(hidden_size):
    grid[i].imshow(np.reshape(params['Wlayer1'][:, i], (32, 32)))  # The AxesGrid object work as a list of axes.
    plt.axis('off')
    
# with default settings, you should get loss < 150 and accuracy > 80%
train_loss, train_accuracy = [], []
valid_loss, valid_accuracy = [], []

for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        yb = yb.astype(int)
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']

    total_acc /= batch_num

    train_loss.append(total_loss/10000)
    train_accuracy.append(total_acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
    
    # Run on validation set per iteration
    valid_y = valid_y.astype(int)  # Ensure labels are integers, if necessary
    h1 = forward(valid_x, params, 'layer1')  # Forward pass through first layer
    probs = forward(h1, params, 'output', softmax)  # Forward pass through output layer

    # Compute loss and accuracy
    loss, acc = compute_loss_and_acc(valid_y, probs)

    # Record validation loss and accuracy
    valid_loss.append(loss)  # Assuming compute_loss_and_acc returns mean loss per example
    valid_accuracy.append(acc)  # Record accuracy as is

# run on validation set and report accuracy! should be above 75%
valid_acc = None
##########################
##### your code here #####
##########################
plt.figure('accuracy')
plt.plot(range(max_iters), train_accuracy, color='g')
plt.plot(range(max_iters), valid_accuracy, color='b')
plt.legend(['train', 'validation'])
plt.show()

plt.figure('loss')
plt.plot(range(max_iters), train_loss, color='g')
plt.plot(range(max_iters), valid_loss, color='b')
plt.legend(['train', 'validation'])
plt.show()

h1 = forward(valid_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################
fig = plt.figure()
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 )

for i in range(hidden_size):
    grid[i].imshow(np.reshape(params['Wlayer1'][:, i], (32, 32)))  # The AxesGrid object work as a list of axes.
    plt.axis('off')

plt.show()


# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))



# compute comfusion matrix here
##########################
##### your code here #####
##########################
def load_data(file_path):
    """
    Load data from a .mat file.

    Args:
    file_path (str): Path to the .mat file.

    Returns:
    tuple: Features and labels from the file.
    """
    data = scipy.io.loadmat(file_path)
    x, y = data[list(data.keys())[-2]], data[list(data.keys())[-1]]
    return x, y

def generate_cm(probs, y):
    """
    Generate a confusion matrix.

    Args:
    probs (ndarray): Probabilities from the network output.
    y (ndarray): True labels.

    Returns:
    ndarray: Confusion matrix.
    """
    # Get predicted labels
    pred_labels = np.argmax(probs, axis=1)
    true_labels = np.argmax(y, axis=1)

    # Create confusion matrix
    confusion_matrix = np.zeros((y.shape[1], y.shape[1]))
    for true, pred in zip(true_labels, pred_labels):
        confusion_matrix[true, pred] += 1

    return confusion_matrix

# Load datasets
train_x, train_y = load_data('../data/nist36_train.mat')
valid_x, valid_y = load_data('../data/nist36_valid.mat')
test_x, test_y = load_data('../data/nist36_test.mat')

# Forward pass through the network for each dataset
h1_train = forward(train_x, params, 'layer1')
train_probs = forward(h1_train, params, 'output', softmax)

h1_valid = forward(valid_x, params, 'layer1')
valid_probs = forward(h1_valid, params, 'output', softmax)

h1_test = forward(test_x, params, 'layer1')
test_probs = forward(h1_test, params, 'output', softmax)

# Generate confusion matrices
train_cm = generate_cm(train_probs, train_y)
valid_cm = generate_cm(valid_probs, valid_y)
test_cm = generate_cm(test_probs, test_y)

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()

# confusion matrix for validation data
confusion_matrix = generate_cm(valid_probs, valid_y)
plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()

# confusion matrix for test data
confusion_matrix = generate_cm(test_probs, test_y)
plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()

