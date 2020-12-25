from scipy.io import loadmat
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

def load_data():
    """
    Use loadmat to load digit_recog.mat
    Return a tuple for our Training_data validation_data test_data
    to make it ready for our neural network 
    a list of 5000 picture that x is a 400_diamentional numpy.ndarray
    and y is a 10 dimentional numpdy.ndarray 
    split them for the first part conditions 
    
    
    """
   # data = loadmat('digit_recog.mat')
    numbers =[]
    train_data = loadmat('digit_recog.mat')['X']
    mat_output = loadmat('digit_recog.mat')["Y"]
    #print (train_data[0])
    
    
    for i in mat_output:
        for j in i:
            numbers.append(j)
    arr = np.array(numbers)     
    a = (train_data,arr)
    train_dataR =[]
    arrR = []
    test_valid1 =[]
    test_valid2 =[]
    test1 = []
    test2 = []
    valid1 = []
    valid2 = []    
    randomlist = random.sample(range(0, 5000), 2500)
    for i in randomlist:
        train_dataR.append(train_data[i])
        arrR.append(arr[i])
    b = (train_dataR,arrR)
    for i in range(5000):
        if i not in randomlist:
            test_valid1.append(train_data[i])
            test_valid2.append(arr[i])
    randomlist1 = random.sample(range(0,2500),1250)
    for i in range(2500):
        if i in randomlist1:
            test1.append(test_valid1[i])
            test2.append(test_valid2[i])
        else:
            valid1.append(test_valid1[i])
            valid2.append(test_valid2[i])
    c = (test1,test2)
    d = (valid1,valid2)         
            
            
        
        
    
            
            
        
  
    training_inputs = [np.reshape(x, (400,1)) for x in b[0]]
    ## the required format for our neural network's input layer
    ## ---
    ##print("hello")
    ##print(type(training_inputs[0][0]))
    training_results = [vectorized_result(y) for y in b[1]]
    ## 
    ## function will convert the digit value in 10-dimensional vector
    ## the required format for our neural network's output layer
    ## ---
    training_data = zip(training_inputs, training_results)
    
    
    #print((valid_data[0]))
    validation_inputs = [np.reshape(x, (400,1)) for x in d[0]]
    
    ## reshaping the validation inputs to 784x1 vector
    ## ---
    validation_data = zip(validation_inputs, d[1])
   
    
    ## test_data:
    test_inputs = [np.reshape(x, (400,1)) for x in c[0]]
    ## reshaping the test inputs to 784x1 vector
    ## ---
    test_data = zip(test_inputs, c[1])
    

    return (training_data, validation_data, test_data)

def vectorized_result(y):
    """
        convert digit to the 10 dimentional that the yth place is 1 and other is equal 0

    """

    e = np.zeros((10,1))

    if y == 10:
        e[0] = 1.0
    else:
        e[y] = 1.0


    return e










class Network:

    def __init__(self, sizes):
        """
        the list sizes refer to the Neural layers and the number of neurons
        """

        self.num_layers = len(sizes)
        ## total number of layers in the network

        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        ## randomly initializing the values for biases

        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        ## randomly initializing the values for weights

    def feedforward(self, a):
        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def GD(self, training_data, test_data, epochs, mini_batch_size, eta):
        """
        Training neural network 
        Eta is our learning rate 
        and after each epoch we will update the bias and weights 
        """

        test_data = list(test_data)
        
        ## in-order to access the data inside it we have to convert it into the list format

        training_data = list(training_data)
        
        ## converting it into list format

        n_test, n_train = len(test_data), len(training_data)
        ## length of test_data and train_data
        epoches =[]
        ## Mini-Batch Stochastic Gradient Descent:
        for j in range(epochs):
            random.shuffle(training_data)
            ## shuffling prevents the any bias during training

            mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            ## making the mini-batches as per mini-batch-size

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
           
            ## the funtion will update the values for weights and biases
            epoches.append(self.evaluate(test_data))
            print(f'Epoch {j}: {self.evaluate(test_data)}/{n_test}')
            ## finding the accuracy after each epoch
            ## plot the accuracy during epochs and calculate the average of them
          
        print(epoches)
        avg = 0
        for i in epoches:
            avg += i
        avg /= len(epoches)
        avg /= n_test
        print("the Avg Accuracy:")
        print(avg)
        xPlot = []
        yPlot = []
        for i in range(len(epoches)):
            xPlot.append(i)
        for i in epoches:
            i /= n_test
            yPlot.append(i)
        plt.plot(xPlot,yPlot)
        plt.savefig('error_3.png')
        plt.clf()            

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini-batch.

        
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        ## nabla_b is the list that will store change in the values of biases
        ## every bias's value is initially initialized as 0

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        ## nabla_w is the list that will store change in the values of weights
        ## every weight's value is initially initialized as 0

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            ## 
            ## delta_nabla_b and delta_nabla_w will contain change in weights and biases respectively

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            ## updating the values in nabla_b list

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            ## updating the values in nabla_w list

        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        ## updating the values of weights by using gradient descent

        self.baises = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        ## updating the values of biases by using gradient descent

    def backprop(self, x, y):
        """
             : ) obviously it is Backpropagation
        """

        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x

        activations = [x]


        zs = []


        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b


            zs.append(z)


            activation = sigmoid(z)


            activations.append(activation)


        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        ## calculating the error in the last layer
        

        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
     
        


        
        for layer in range(2, self.num_layers):
            z = zs[-layer]
         

            sp = sigmoid_prime(z)
            

            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
       

            delta_nabla_b[-layer] = delta
            delta_nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
           

        return (delta_nabla_b, delta_nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outputs the correct result
        """

        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]

        return sum(int(x == y) for (x,y) in test_results)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

def cost_derivative(output_activations, y):
    """Derivative of cost function."""
    return (output_activations - y)





training_data, validation_data, test_data = load_data()


net = Network([400, 26, 10])


net.GD(training_data, test_data, epochs=40, mini_batch_size=10, eta=2.0)

