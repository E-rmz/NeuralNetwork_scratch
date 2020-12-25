
import numpy as np




train_data = np.array([[1,0,0,1,0,0,1,1,1,1],[0,1,0,0,1,0,0,1,0,1]])
 


weights = np.random.normal(size =(10))
# Weights and Bias in one Ndarray

output_train_data = np.array([1,-1])
## L >> 1 and I >> 0


noisy_tests = [ [0,1,0,0,1,0,0,0,0,1] , 
                 [1,0,0,1,1,0,0,1,0,1] ,
                 [1,1,0,1,0,1,1,1,1,1] , 
                 [1,1,0,0,1,0,0,1,0,1] , 
                 [0,1,0,1,0,0,0,1,0,1] , 
                 [0,1,0,0,1,0,1,1,0,1]]



eta=0.1
Y = 0
error = 1


"""
algorithm definition
"""

while(error!=0):
    error = 0
    for i in range(2):
        Y = np.dot(train_data[i],weights)
        r = eta * (output_train_data[i]-np.sign(Y))
        delta_w = r*train_data[i]
        weights = weights+delta_w   
        error+= 0.5*(output_train_data[i] - np.sign(Y))**2   

        
"""
Vision of pixels
"""


def pixels(example):
    print("{}  {}  {}".format(example[0],example[1],example[2]))
    print("{}  {}  {}".format(example[3],example[4],example[5]))
    print("{}  {}  {}".format(example[6],example[7],example[8]))



"""
last step and test it ! 
"""

def results(test):
    for item in test :
        predict = np.sign(np.dot(item,weights))
        pixels(item)
        if(predict==1):
            print("prediction is L")
        else:
            print("prediction is I")
        print("\n*************\n*************")




results(train_data) 
results(noisy_tests)   
print("weights are :")    
print(weights)
      

