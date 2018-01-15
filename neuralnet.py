import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import ndimage, misc
import pickle


class NeuralNet:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.W = []
        self.b = []
        self.learning_rate = 0.0075
        self.iterations = 1500
        self.layer_dims = [train_x.shape[0], 1]


def load_data():
    # load training dataset
    train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    train_set_x = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])

    # load test dataset
    test_dataset = h5py.File("datasets/test_catvnoncat.h5", "r")
    test_set_x = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])

    # shape y sets to be consistent with the rest of the data
    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))

    # flatten datasets
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T
    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T

    # standardize datasets
    train_set_x = train_set_x / 255.
    test_set_x = test_set_x / 255.

    return train_set_x, train_set_y, test_set_x, test_set_y


def initialize_parameters(layer_dims):
    """Initialize the parameters to have random values.

    arguments:
    layer_dims -- number of nodes in each layer of the network

    returns:
    W -- python array of weight parameter matrices
    b -- python array of bias parameter vectors
    """
    np.random.seed(1)
    L = len(layer_dims)

    W = []
    b = []
    for l in range(1, L):
        rand_W = np.random.randn(layer_dims[l], layer_dims[l-1])
        W.append(rand_W / np.sqrt(layer_dims[l-1]))
        b.append(np.zeros((layer_dims[l], 1)))

    return W, b


def sigmoid(Z):
    """Call sigmoid activation function on linear output from previous layer."""
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    """Call ReLU activation function on linear output from previous layer."""
    A = np.maximum(0, Z)
    return A


def sigmoid_derivative(dA, Z):
    """Use the derivative of the sigmoid function to find the gradient of the
    cost function with respect to Z.
    """
    s = sigmoid(Z)
    dZ = dA * s * (1 - s)
    return dZ


def relu_derivative(dA, Z):
    """Use the derivative of the ReLU function find the gradient of the cost
    function with respect to Z.
    """
    dZ = np.array(dA)
    dZ[Z <= 0] = 0
    return dZ


def forward_step(A_prev, W, b, activation):
    """Step forward through one layer of the network, computing linear output Z
    using the previous layer's activation output, then computing non-linear
    activation output A for this layer.

    arguments:
    A_prev -- previous activation layer
    W -- parameter matrix for this layer
    b -- bias vector for this layer
    activation -- string specifying activation function

    returns:
    A -- current activation layer
    cache -- values to be used for backward propagation
    """
    Z = W.dot(A_prev) + b

    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)

    cache = (A_prev, Z, W, b)

    return A, cache


def forward_propagation(X, W, b):
    """Propagate forward through all layers of the network. Use the ReLU
    activation function for each layer except for the output layer. Use the
    sigmoid activation function for the activation layer. Append each cache
    returned from the forward step to a list.

    arguments:
    X -- input matrix of training dataset
    W -- python list of weight parameter matrices
    b -- python list of bias parameter vectors

    returns:
    AL -- final output vector for the network
    caches -- python list of cache values used for backward propagation
    """
    caches = []
    A = X
    L = len(W) - 1

    for l in range(0, L):
        A_prev = A
        A, cache = forward_step(A_prev, W[l], b[l], "relu")
        caches.append(cache)

    AL, cache = forward_step(A, W[L], b[L], "sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """Compute cost of the output of the network given its current
    parameters.
    """
    m = Y.shape[1]
    cost = (-np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T)) / m
    cost = np.squeeze(cost)

    return cost


def backward_step(dA, cache, activation):
    """Step backwards through one layer of the network, computing gradient of
    the cost function with respect to the previous layer's activation output A.

    arguments:
    dA -- gradient of the cost with respect to A for this layer
    cache -- cached values from forward step
    activation -- string specifying activation function

    returns:
    dA_prev -- gradient of the cost with respect to A for previous layer
    dW -- weight gradient matrix
    db -- bias gradient matrix
    """
    A_prev, Z, W, b = cache
    m = A_prev.shape[1]

    if activation == "relu":
        dZ = relu_derivative(dA, Z)
    elif activation == "sigmoid":
        dZ = sigmoid_derivative(dA, Z)

    dW = np.dot(dZ,A_prev.T) / m
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    """Propagate backwards through all layers of the neural network to find the
    gradients of all weight and bias parameters. Use the derivative of the ReLU
    function for each layer except for the activation layer. Use the derivative
    of the sigmoid function for the activation layer.

    arguments:
    AL -- final output vector for the network
    Y -- output vector of training dataset
    caches -- all cached from forward propagation

    returns:
    dW -- python array of weight gradient matrices
    db -- python array of bias gradient matrices
    """
    dW = []
    db = []
    L = len(caches)
    m = Y.shape[1]

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    dA_prev, dW_temp, db_temp = backward_step(dAL, caches[L-1], "sigmoid")
    dW.append(dW_temp)
    db.append(db_temp)

    for l in reversed(range(L-1)):
        dA_prev, dW_temp, db_temp = backward_step(dA_prev, caches[l], "relu")
        dW = [dW_temp] + dW
        db = [db_temp] + db

    return dW, db


def update_parameters(W, b, dW, db, learning_rate):
    """Update all parameters of the network using the gradients of the cost
    function with respect to the parameters in an attempt to step toward a
    global minimum of the cost function. Learning rate determines how quickly
    the parameters are updated.
    """
    L = len(W)
    for l in range(L):
        W[l] = W[l] - learning_rate * dW[l]
        b[l] = b[l] - learning_rate * db[l]

    return W, b


def model(X, Y, layer_dims, learning_rate = 0.0075, iterations = 2500):
    """For each iteration, propagate forward through the network,
    compute the cost, propagate backward through the network, and then finally
    update the parameters based on the gradients computed via back propagation.

    arguments:
    X -- input matrix of the dataset
    Y -- output matrix of the dataset
    layer_dims -- lengths of each layer in the network
    learning_rate -- rate at which parameters are updated
    iterations -- number of training iterations

    returns:
    W -- python array of weight parameter matrices
    b -- python array of bias parameter vectors
    """
    costs = []

    W, b = initialize_parameters(layer_dims)

    for i in range(0, iterations):
        AL, caches = forward_propagation(X, W, b)
        cost = compute_cost(AL, Y)
        dW, db = backward_propagation(AL, Y, caches)
        W, b = update_parameters(W, b, dW, db, learning_rate)

        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations")
    plt.title("Learning rate: " + str(learning_rate))
    plt.show()

    return W, b


def predict(X, Y, W, b):
    """Predict output vector Y for input matrix X and return accuracy."""
    m = X.shape[1]
    n = len(W)

    probabilities, caches = forward_propagation(X, W, b)
    predictions = np.round(probabilities)

    accuracy = str(np.sum((predictions == Y) / m))
    return accuracy


def predict_image(W, b, fname):
    """Predict output for custom image."""
    fname = "images/" + (fname or "my_image.jpg")
    image = np.array(ndimage.imread(fname, flatten=False))
    image = misc.imresize(image, size=(64,64)).reshape((64*64*3,1))

    probability, cache = forward_propagation(image, W, b)

    return np.squeeze(probability)


def save_parameters(W, b, layer_dims, learning_rate, iterations):
    """Save parameters and hyperparameters to a save file."""
    hyperparameters = {
        "layer_dims": layer_dims,
        "learning_rate": learning_rate,
        "iterations": iterations
    }

    with open("save/hyperparameters.pickle", "wb") as handle:
        pickle.dump(hyperparameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("save/parameters.npz", "wb") as handle:
        np.savez(handle, W=W, b=b)


def load_parameters():
    """Load parameters and hyperparameters from a save file."""
    with open("save/parameters.npz", "rb") as handle:
        parameters = np.load(handle)
        W = parameters["W"]
        b = parameters["b"]

    with open('save/hyperparameters.pickle', 'rb') as handle:
        hyperparameters = pickle.load(handle)
        layer_dims = hyperparameters["layer_dims"]
        learning_rate = hyperparameters["learning_rate"]
        iterations = hyperparameters["iterations"]

    return W, b, layer_dims, learning_rate, iterations
