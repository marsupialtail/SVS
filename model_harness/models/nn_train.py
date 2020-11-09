import numpy as np
from util import *
import time
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self,num_inputs,num_outputs,hidden_layers = 1,units_per_layer = 10):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.units_per_layer = units_per_layer
        self.hidden_layers = hidden_layers
        self.weights = [np.random.normal(scale=np.sqrt(2 / num_inputs),size=(num_inputs,units_per_layer))]
        self.biases = [np.random.normal(scale=np.sqrt(2 / num_inputs),size=(1,units_per_layer))]
        for i in range(hidden_layers-1):
            self.weights.append(np.random.normal(scale=np.sqrt(2 / units_per_layer),size=(units_per_layer,units_per_layer)))
            self.biases.append(np.random.normal(scale=np.sqrt(2 / num_inputs), size=(1, units_per_layer)))

        self.weights.append(np.random.normal(scale=np.sqrt(2 / units_per_layer),size=(units_per_layer,num_outputs)))
        self.biases.append(np.zeros((1,num_outputs)))

        self.approx_1 = None
        self.approx_2 = None

        self.out = None
        self.angles = {i:[] for i in range(hidden_layers + 1)}
        self.mm1_timer = 0
        self.mm2_timer = 0

    def forward(self,epoch_num,batch_in):

        self.batch_size = batch_in.shape[0]
        self.epoch_num = epoch_num

        #batch_in must be a batch_size * num_inputs array
        #batch_out must be a batch_size * num_outputs array

        self.hidden_values = [batch_in.copy()]
        self.activations = []
        for i in range(1,len(self.weights)+1):
            self.activations.append(np.matmul(self.hidden_values[i-1],self.weights[i-1])+np.matmul(np.ones((self.batch_size,1)),self.biases[i-1]))

            if i != len(self.weights):
                self.hidden_values.append((relu(self.activations[-1])))
            else:
                self.hidden_values.append(self.activations[-1])

        self.out = softmax(self.hidden_values[-1])

    def calc_gradients(self,batch_out,approx_1 = False, approx_2 = False):

        if self.out is None:
            print("Must run forward pass first")
        else:

            self.approx_1 = approx_1
            self.approx_2 = approx_2

            #self.gradient will be a list where the first element
            #is the gradient of self.i2h, then self.h2h[0] ... self.h2o
            # intialized to zero array to enforce dimension correctness
            self.weight_gradients = []
            self.bias_gradients = []
            for i in range(len(self.weights)):
                self.weight_gradients.append(np.zeros((self.weights[i].shape[0],self.weights[i].shape[1])))
                self.bias_gradients.append(np.zeros((self.biases[i].shape[0],self.biases[i].shape[1])))
            errors = softmax_prime(self.out.copy(),batch_out) / self.batch_size
            for i in range(len(self.weight_gradients)-1,-1,-1):

                start = time.time()

                if not self.approx_1:
                    self.weight_gradients[i] = np.matmul(self.hidden_values[i].T,errors)
                else:
                   self.weight_gradients[i] = approximate_matmul1(errors.T,self.hidden_values[i]).T

                self.mm1_timer += time.time() - start

                #actual_gradient = np.matmul(self.hidden_values[i].T,errors)
                #self.angles[i].append(calc_matrix_angle(self.weight_gradients[i],actual_gradient))

                self.bias_gradients[i] = np.sum(errors,axis=0)

                start = time.time()

                if i > 0:
                    if i == len(self.weight_gradients)-1:
                        errors = np.matmul(errors,self.weights[i].T) * relu(self.activations[i-1])
                    else:
                        if self.approx_2:
                            errors = approximate_matmul2(self.weights[i],errors.T).T * relu(self.activations[i-1])
                        else:
                            errors = np.matmul(self.weights[i],errors.T).T * relu(self.activations[i-1])

                self.mm2_timer += time.time() - start



    def make_step(self,learning_rate):

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate * self.weight_gradients[i]
            self.biases[i] = self.biases[i] - learning_rate * self.bias_gradients[i]

            if np.min(self.biases[i]) < -1000 or np.max(self.biases[i]) > 1000:
                print(self.epoch_num)
                print(self.biases[i])

    # get shape * 78

class Trainer:

    def __init__(self,batch_size):

        self.train_data, self.train_labels, self.validation_data, self.validation_labels = read_mnist()
        self.train_data = np.array(self.train_data)
        self.train_labels = np.expand_dims(np.array(self.train_labels),1)

        self.batch_size = batch_size
        self.learning_rate = 0.02
        self.net1 = NeuralNetwork(784,10,2,400)
        self.net2 = NeuralNetwork(784,10,2,400)

        self.train_accs1 = []
        self.train_accs2 = []
        self.val_accs1 = []
        self.val_accs2 = []

    def train_epoch(self,epoch_num):

        self.train_data, self.train_labels = shuffle_in_unison(self.train_data, self.train_labels)

        self.forward_timer1 = 0
        self.backward_timer1 = 0

        self.forward_timer2 = 0
        self.backward_timer2 = 0

        num_batches = len(self.train_labels) // self.batch_size
        for i in range(num_batches):
            batch_in = self.train_data[i*self.batch_size:(i+1)*self.batch_size]
            batch_out = self.train_labels[i*self.batch_size:(i+1)*self.batch_size]

            self.start = time.time()
            self.net1.forward(i,batch_in)
            self.forward_timer1 += time.time() - self.start

            self.start = time.time()
            if epoch_num % 2 == 1:
                self.net1.calc_gradients(batch_out,True,True)
            else:
                self.net1.calc_gradients(batch_out,False,False)

            self.net1.make_step(self.learning_rate)
            self.backward_timer1 += time.time() - self.start

            self.start = time.time()
            self.net2.forward(i, batch_in)
            self.forward_timer2 += time.time() - self.start

            self.start = time.time()
            self.net2.calc_gradients(batch_out,False,False)
            self.net2.make_step(self.learning_rate)
            self.backward_timer2 += time.time() - self.start

        print(self.net1.mm1_timer, self.net1.mm2_timer)
        print(self.net2.mm1_timer, self.net2.mm2_timer)


        #for i in self.net.angles:
        #    plt.hist(self.net.angles[i])
        #    plt.show()

        print(self.forward_timer1, self.backward_timer1)
        print(self.forward_timer2, self.backward_timer2)

        #print(self.net.weights)
        #print(self.net.biases)
        self.net1.forward(0,self.train_data)
        self.net2.forward(0,self.train_data)

        acc1 = np.sum(np.argmax(self.net1.out,axis=1)==self.train_labels.squeeze())/len(self.train_labels)
        acc2 = np.sum(np.argmax(self.net2.out,axis=1)==self.train_labels.squeeze())/len(self.train_labels)

        self.train_accs1.append(acc1)
        self.train_accs2.append(acc2)
        print(acc1,acc2)

        self.net1.forward(0, self.validation_data)
        self.net2.forward(0, self.validation_data)

        acc1 = np.sum(np.argmax(self.net1.out, axis=1) == self.validation_labels.squeeze()) / len(self.validation_labels)
        acc2 = np.sum(np.argmax(self.net2.out, axis=1) == self.validation_labels.squeeze()) / len(self.validation_labels)

        self.val_accs1.append(acc1)
        self.val_accs2.append(acc2)
        print(acc1, acc2)

        #print(training_error)

brian = Trainer(30)
for i in range(10):
    brian.train_epoch(i)

plt.plot(brian.train_accs1,'b')
plt.plot(brian.train_accs2,'g')

plt.plot(brian.val_accs1,'k')
plt.plot(brian.val_accs2,'r')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()