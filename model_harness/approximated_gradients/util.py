import numpy as np
import pickle
import gzip

def read_mnist():

    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    return training_data[0], training_data[1], validation_data[0], validation_data[1]

def relu(num):
    return np.maximum(num,0)

def sigmoid(num):
    return 1/(1+np.exp(-num))

def sigmoid_prime(num):
    return np.multiply(sigmoid(num),(1-sigmoid(num)))

def softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps,axis=1,keepdims=True)

def softmax_prime(x,labels):

    # x: k by t
    # labels: k * 1

    n = x.shape[0]
    x[np.arange(n),labels.astype(int).squeeze()] -= 1
    return x

def shuffle_in_unison(a, b):
    permutation = np.random.permutation(len(a))
    shuffled_a = []
    shuffled_b = []
    for index in permutation:
        shuffled_a.append(a[index])
        shuffled_b.append(b[index])
    return np.array(shuffled_a), np.array(shuffled_b)

def approximate_matmul1(a,b):

    b_max = np.argmax(b,axis=0)

    return a[:,b_max]

def approximate_matmul2(a,b):

    b_max = np.argmax(np.abs(b),axis=0)
    b_sum = np.sum(b,axis=0)
    #b_sign = np.sign(b[b_max,np.arange(b.shape[1])])
    #b_sign = np.matmul(np.ones((a.shape[0],1)),np.expand_dims(b_sign,0))
    return a[:,b_max] * np.expand_dims(b_sum,0) #* b_sign

def calc_matrix_angle(a,b):

    return np.sum(a*b) / np.linalg.norm(a) / np.linalg.norm(b)