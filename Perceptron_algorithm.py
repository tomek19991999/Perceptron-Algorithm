from csv import reader
from math import sqrt
import random
from collections import Counter

"""
FORMAT OF FILE
1. variance of Wavelet Transformed image (continuous) 
2. skewness of Wavelet Transformed image (continuous)
3. curtosis of Wavelet Transformed image (continuous)
4. entropy of image (continuous)
5. class (integer) 
"""

#Loading from CSV file
def loading_txt (file):
    data=list()
    with open(file, 'r') as file:
        csvreader = reader(file)
        for row in csvreader:
            if not row:
                continue
            data.append(row)

    #make float data (before, we had string data)
    for row in data:
        for i in range(len(data[0])):
            row[i]=float(row[i])
            #print(row[i])
    return data

# Make a prediction with weights
def predict(row, weights):
    activation = 0 
    if(len(row)==5):
        for i in range(len(row)-1):
            activation += weights[i + 1] * row[i] # weight * column in data row
    else:
        for i in range(len(row)):
            activation += weights[i + 1] * row[i] # weight * column in data row
    activation += weights[0] #bias
    if activation >= 0.0:
        return 1.0
    else:
        return -1.0  


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train_dataset, l_rate, n_epoch, weights):

    #start calculating
    for epoch in range(n_epoch): #calculate epoch times
        sum_error = 0.0
        for row in train_dataset:
            prediction = predict(row, weights)
            if (prediction ==-1):
                prediction=0
            error = row[-1] - prediction 

            #if expected!=predicted, update weights
            if error**2 == 1:
                sum_error += error**2 
                weights[0] = weights[0] + l_rate * error # bias(t+1) = bias(t) + learning_rate * (expected(t) - predicted(t))
                for i in range(len(row)-1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i] #w(t+1)= w(t) + learning_rate * (expected(t) - predicted(t)) * x(t)
        print('>epoch=%d, l_rate=%.3f, sum_error=%.3f' % (epoch, l_rate, sum_error))
    print(weights)
    return weights

def validation_data_test(train_dataset, l_rate, n_epoch, weights):
    #creating list for confusion matrix
    arr = [[0 for i in range(2)] for j in range(len(train_dataset))] #creating list for OK or NOT_OK results. 2columns, a lot of rows [EXPECTED,PREDICTED]
    matrix=list()
    matrix=[[0 for i in range(2)] for j in range(2)]

    for row in train_dataset:
        prediction = predict(row, weights)
        if (prediction ==-1):
            prediction=0
        #take data into confusion matrix
        if row[-1]==0 and prediction==0:
            matrix[0][0]+=1 #tp_counter
        elif row[-1]!=0 and prediction==0:
            matrix[0][1]+=1 #fp_counter
        elif row[-1]==0 and prediction!=0:
            matrix[1][0]+=1 #fn_counter
        elif row[-1]!=0 and prediction!=0:
            matrix[1][1]+=1 #tn_counter

    print ("\nLEGEND:","\n   0  1""\n0 TP|FP\n  -----\n1 FN|TN")
    print("\n",matrix[0][0],"|",matrix[0][1],"\n-------\n",matrix[1][0],"|",matrix[1][1],"\n\n")


def test_data_test(train_dataset, l_rate, n_epoch, weights):
    for row in train_dataset:
        prediction = predict(row, weights)
        if (prediction ==-1):
            prediction=0
        print(row,"VALUE:", prediction)


def perceptron_banknote_authentication(learning_rate, n_epoch,weights):

    training_data=list()
    training_data=loading_txt("training_data.txt")
    validation_data=list()
    validation_data=loading_txt("validation_data.txt")
    test_data=list()
    test_data=loading_txt("test_data.txt")
    #delete names in last column
    for j in test_data:
        del j[4]
    #print(data)
    weights = train_weights(training_data, learning_rate, n_epoch,weights)
    validation_data_test(validation_data, learning_rate, n_epoch, weights)
    test_data_test(test_data, learning_rate, n_epoch, weights)

# test predictions
l_rate = 0.1
n_epoch = 20
weights=[0.0,0.0,0.0,0.0,0.0] #weights[0] is bias

perceptron_banknote_authentication(l_rate, n_epoch,weights)
