import numpy as np
import tensorflow as tf
import math
import logging
logging.basicConfig(level=logging.DEBUG)
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import KFold
import pandas as pd
#import winsound


x = pd.read_csv("predx_for_regression.csv", header=0)
x_array = np.asarray(x, dtype = "float")
x_arrayt=x_array

y = pd.read_csv("predy_for_regression.csv", header=0)
y_array = np.asarray(y, dtype = "float")
y_arrayt=y_array

whole_data=np.concatenate((x_arrayt, y_arrayt),axis=1)


angle = pd.read_csv("angle.csv", header=0)
angle_array = np.asarray(angle, dtype = "float")
angle_arrayt=angle_array.transpose()


#Network parameters
n_hidden1 = 35
n_input = 98
n_output = 1

#Learning parameters
learning_constant = 0.006
number_epochs = 4000
batch_size = 500

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

#DEFINING WEIGHTS AND BIASES

#Biases first hidden layer
b1 = tf.Variable(tf.random_normal([n_hidden1],mean = 0, stddev = 0.7))

#Biases output layer
b3 = tf.Variable(tf.random_normal([n_output],mean = 0, stddev = 0.7))


#Weights connecting input layer with first hidden layer
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1],mean = 0, stddev = 0.7))

#Weights connecting first hidden layer with output layer
w5 = tf.Variable(tf.random_normal([n_hidden1, n_output],mean = 0, stddev = 0.7))



#The incoming data given to the network is input_d
def multilayer_perceptron(input_d):
    #Task of neurons of first hidden layer
    layer_1 = tf.nn.relu(tf.add(tf.matmul(input_d, w1), b1))    
    #Task of neurons of output layer
    out_layer = tf.add(tf.matmul(layer_1, w5),b3)
    
    return out_layer

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))

#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

label=angle_array#+1e-50-1e-50
batch_x=(whole_data-200)/2000
temp=np.array([angle_array[:,0]])
batch_y=temp.transpose()
#Normalise the labels
batch_y = (batch_y - np.mean(batch_y, axis=0)) / np.std(batch_y, axis=0)
label_train=label
label_test=label


#Initialisse the cross-validation
k_fold = 10
kf = KFold(n_splits=k_fold)
print(kf)
kf.get_n_splits(whole_data)
average_fold_error = np.zeros(k_fold)
fold_number = 0

#Start the loop for cross-validation
for train_index, test_index in kf.split(whole_data):
    print("Fold: %d" % (fold_number))
    print("Train index", end="")
    print(train_index)
    print("Test index", end="")
    print(test_index)
    
    batch_x_train, batch_x_test = batch_x[train_index], batch_x[test_index]
    batch_y_train, batch_y_test = batch_y[train_index], batch_y[test_index]
    
    #create a session
    with tf.Session() as sess:
        sess.run(init)
        #Training epoch
        for epoch in range(number_epochs):
            #Run the optimizer feeding the network with the batch
            sess.run(optimizer, feed_dict={X: batch_x_train, Y: batch_y_train})
            #Display the epoch
            if epoch % 100 == 0 and epoch>10:
                print("Epoch:", '%d' % (epoch))
                print("Error_train:", loss_op.eval({X: batch_x_train, Y: batch_y_train}) )
    
    
        # Test model
        pred = (neural_network)# Apply softmax to logits
        accuracy=tf.keras.losses.MSE(pred,Y)
        print("Error_test:", np.square(accuracy.eval({X: batch_x_test, Y: batch_y_test})).mean() )
        #tf.keras.evaluate(pred,batch_x)
        average_fold_error[fold_number] = np.square(accuracy.eval({X: batch_x_test, Y: batch_y_test})).mean()
        fold_number += 1
        print("Prediction:", pred.eval({X: batch_x_test}))
        print("Actual values:", batch_y_test)
    
        output=neural_network.eval({X: batch_x_test})
        
        plt.plot(batch_y_test, 'r', output, 'b')
        plt.ylabel('Pitch of the rotation')
        plt.show()
       
        #Plot with reversed overlay as too see overfitting easier
        plt.plot(output, 'b', batch_y_test, 'r')
        plt.ylabel('Pitch of the rotation')
        plt.show()
        
        
        #We only need 1 output as to not have too many csv files.
        if fold_number == 1:
            df = DataFrame(output)
            df2 = DataFrame(batch_y_test)
            export_csv = df.to_csv ('fold1_output.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
            export_csv = df.to_csv ('Normalised_output_fold1.csv', index = None, header = True)
    
        
        correct_prediction = tf.math.subtract((pred), (Y))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: batch_x, Y: batch_y}))
#Save the error of all our folds           
folds = DataFrame(average_fold_error)
export_csv = folds.to_csv ('fold_errors.csv', index = None, header = True)      
print(average_fold_error)
print(np.mean(average_fold_error))
#winsound.Beep(1000,1000)       