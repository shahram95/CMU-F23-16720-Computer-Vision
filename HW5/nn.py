import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    if in_size <= 0 or out_size <= 0:
        raise ValueError("Input and output sizes must be positive")

    # Calculate the bounds for Xavier initialization
    limit = np.sqrt(6 / (in_size + out_size))

    # Initialize weights uniformly within [-limit, limit]
    W = np.random.uniform(-limit, limit, (in_size, out_size))
    # Initialize biases to zeros
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    ##########################
    ##### your code here #####
    ##########################

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################
    pre_act = np.dot(X, W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    # Shift input for numerical stability by subtracting the max in each row
    shift_x = x - np.max(x, axis=1, keepdims=True)
    
    # Calculate the exponentials and normalize each row
    exps = np.exp(shift_x)
    res = exps / np.sum(exps, axis=1, keepdims=True)

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################
    # Check for empty input to avoid division by zero
    if y.shape[0] == 0:
        return 0, 0

    # Compute Categorical Cross-Entropy Loss
    # Adding a small constant for numerical stability to avoid log(0)
    probs_stable = np.clip(probs, 1e-8, 1 - 1e-8)
    loss = -np.sum(y * np.log(probs_stable)) / y.shape[0]

    # Compute Accuracy
    correct_predictions = np.sum(np.argmax(probs, axis=1) == np.argmax(y, axis=1))
    acc = correct_predictions / y.shape[0]

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################
    # Apply the derivative of the activation function
    delta_scaled = delta * activation_deriv(post_act)

    # Compute gradients
    grad_W = np.dot(X.T, delta_scaled)
    grad_b = np.sum(delta_scaled, axis=0)
    grad_X = np.dot(delta_scaled, W.T)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################
    if len(x) != len(y):
        raise ValueError("Input features and target values must have the same number of samples.")
    
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_x = x[batch_indices]
        batch_y = y[batch_indices]
        batches.append((batch_x, batch_y))
    return batches
