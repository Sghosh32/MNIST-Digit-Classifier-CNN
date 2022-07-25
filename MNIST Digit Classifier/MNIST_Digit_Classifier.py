# Importing necesaary libraries
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import gzip

def load_data(x_train, y_train, x_test, y_test, image_size = 28, train_size = 60000, test_size = 10000):

    # Training Images
    f = gzip.open(x_train, 'r')
    f.read(16)
    buffer = f.read(image_size * image_size * train_size)
    data = np.frombuffer(buffer,dtype = np.uint8).astype(np.float32)
    x_train = data.reshape(train_size, image_size * image_size, 1)

    # Training Labels
    f = gzip.open(y_train, 'r')
    f.read(16)
    y_train = []
    for i in range(train_size):
        buffer = f.read(1)
        labels = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
        y_train.append(labels)
    
    # Test Images
    f = gzip.open(x_test, 'r')
    f.read(16)
    buffer = f.read(image_size * image_size * test_size)
    data = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
    x_test = data.reshape(test_size, image_size * image_size, 1)

    # Test Labels
    f = gzip.open(y_test, 'r')
    f.read(16)
    y_test = []
    for i in range(test_size):
        buffer = f.read(1)
        labels = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
        y_test.append(labels)

    return x_train, y_train, x_test, y_test

def train_set(x_train, y_train, train_size = 60000):
    X = x_train.T.squeeze()
    print(np.shape(y_train))
    Y = np.zeros((10, train_size))
    print(np.shape(Y))
    return X, Y
        
def mini_batch(X, Y, train_size = 60000, batch_size = 32):
    batch_num = train_size // batch_size
    X = np.hsplit(X, batch_num)
    Y = np.hsplit(Y, batch_num)
    return X, Y, batch_num
    
def prime_ReLU(Z):
    return Z > 0

def ReLU(Z):
    return np.maximum(Z, 0)

def Softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis = 0)

def prime_Softmax(Z):
    return (Softmax(Z) * (1 - Softmax(Z)))

def loss_function(A, Y):
    print(np.shape(A), np.shape(Y))
    value = (-1 * Y * np.log(A)) + ((1 - Y) * np.log(1 - A))
    print(value)
    return(value)

def hyperparameters():
    L = 4
    n = [784, 9, 5, 10]
    m = 32
    W = [[]]
    B = [[]]

    for l in range(1,L):
        Wl = np.random.rand(n[l], n[l-1]) * 0.01
        Bl = np.zeros((n[l], 1), dtype = float)
        W.append(Wl)
        B.append(Bl)

        return L, n, W, B, m

def reset_ZA(L, n, Xt, batch_size = 32):
    Z = []
    Z.append(Xt)
    A = []
    A.append(Xt)
    print(Z[0], np.shape(Xt))
    for l in range(1,L):
        Zl = np.zeros((n[l], batch_size), dtype = float)
        Al = np.zeros((n[l], batch_size), dtype = float)
        Z.append(Zl)
        A.append(Al)
    return Z, A

def forward_propagation(L, n, batchsize, W, B, Z, A):
    for i in range(1, len(Z)):
        Z[i] = np.dot(W[i], A[i-1]).reshape(n[i], batch_size) + B[i]
        if(i == L-1):
            A[i] = Softmax(Z[i])
        else:
            A[i] = ReLU(Z[i])
    
    return Z, A

def cost_function(Y_hat, Y):
    batch_size = Y_hat.shape[1]
    cost = (np.dot(Y, np.log(Y_hat).T) + np.dot(1-Y, np.log(1-Y_hat).T)) * (-1 / batch_size)
    return np.squeeze(cost)

def accuracy(Y_hat, Y):
    prob = np.copy(Y_hat)
    prob[prob > 0.5] = 1
    prob[prob < 0.5] = 0
    return (prob == Y).all(axis = 0).mean()

def backward_propagation(A, Z, Y, W, B, L, n):
    batch_size = Y.shape[1]
    dA = []
    da = -(np.divide(Y, A[L-1]) - np.divide(1-Y, 1-A[L-1]))
    dA.append(da)
    dZ = [[]]
    dW = [[]]
    dB = [[]]

    for i in range(L-1, 0, -1):
        if(i == L-1):
            dz = A[i] - Y
        else:
            dz = np.dot(W[i+1].T, dZ[0]) * prime_ReLU(Z[i])

    for i in range(1, L):
        dw = np.dot(dZ[i-1], A[i-1].T) / batch_size
        db = np.sum(dZ[i-1], axis = 1, keepdims = True) / batch_size

        dW.append(dw)
        dB.append(db)

    return dW, dB

def Neural_Network():
    L, n, W, B, batch_size = hyperparameters()

    x_train, y_train, y_test, x_test = load_data(x_train = 'train-images-idx3-ubyte.gz', y_train = 'train-labels-idx1-ubyte.gz', y_test = 't10k-labels-idx1-ubyte.gz', x_test = 't10k-images-idx3-ubyte.gz')
    X, Y = train_set(x_train, y_train)
    _, Batch = (np.shape(X))
    mean = np.sum(X, axis = 1).reshape(784, 1) / Batch

    batch_num, X, Y = mini_batch(X, Y, train_size = 60000, batch_size = 32)
    cost_history = []
    accuracy_history = []

    epoch_no = 100
    for epoch in range(epoch_no):
        for i in range(1875):
            Xt = X[i]
            Yt = Y[i]

            Z, A = reset_ZA(L, n, Xt, batch_size)

            Z, A = forward_propagation(L, n, batch_size, W, B, Z, A)

            J = cost_function(Y_hat = A[L-1], Y = Yt)
            cost_history.append(J)
            Accuracy = accuracy(Y_hat = A[L-1], Y = Yt)
            accuracy_history.append(accuracy)

            dW, dB = back_propagation(A, Z, Yt, W, B, L, n)

            alpha = 0.01
            for i in range(1, L):
                W[i] = W[i] - alpha * dW[i]
                B[i] = B[i] - alpha * dB[i]

        return W, B, cost_history, accuracy_history

W, B, cost, acc = Neural_Network()
plt.plot([i for i in range(len(acc))], acc)
plt.plot([i for i in range(len(cost))], cost)
print(np.sum(acc)/len(acc))
plt.show()
print(acc)
A = [0]
beta = 0.98
for i in range(len(acc)):
    A.append(beta*A[i] + (1-beta)*acc[i])
plt.plot([i for i in range(len(A))], A)
plt.show()



