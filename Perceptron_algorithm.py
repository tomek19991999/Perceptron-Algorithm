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
    activation = 0 #bias
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i] 
    activation += weights[0] #bias
    if activation >= 0.0:
        return 1.0
    else:
        return 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train_dataset, l_rate, n_epoch, weights):
    #creating list for confusion matrix
    arr = [[0 for i in range(2)] for j in range(len(train_dataset))] #creating list for OK or NOT_OK results. 2columns, a lot of rows [EXPECTED,PREDICTED]
    matrix=list()
    matrix=[[0 for i in range(2)] for j in range(2)]

    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train_dataset:
            prediction = predict(row, weights)
            error = row[-1] - prediction #error = expected - predicted. 0-OK   1-BAD
            
            #at last epoch, take data into confusion matrix
            if(epoch==n_epoch-1):
                if row[-1]==0 and prediction==0:
                    matrix[0][0]+=1 #tp_counter
                elif row[-1]!=0 and prediction==0:
                    matrix[0][1]+=1 #fp_counter
                elif row[-1]==0 and prediction!=0:
                    matrix[1][0]+=1 #fn_counter
                elif row[-1]!=0 and prediction!=0:
                    matrix[1][1]+=1 #tn_counter
            
            #if expected!=predicted, update weights
            if error**2 == 1:
                sum_error += error**2 
                weights[0] = weights[0] + l_rate * error # bias(t+1) = bias(t) + learning_rate * (expected(t) - predicted(t))
                for i in range(len(row)-1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i] #w(t+1)= w(t) + learning_rate * (expected(t) - predicted(t)) * x(t)
        print('>epoch=%d, l_rate=%.3f, sum_error=%.3f' % (epoch, l_rate, sum_error))
    print(weights)
    print ("\nLEGEND:","\n   0  1""\n0 TP|FP\n  -----\n1 FN|TN")
    print("\n",matrix[0][0],"|",matrix[0][1],"\n-------\n",matrix[1][0],"|",matrix[1][1])

    return weights

def true_positive_false_positive_confusion_matrix(result,iris_name):
    matrix=list()
    matrix=[[0 for i in range(2)] for j in range(2)]
    print ("\nFor: ",iris_name, "\nTP|FP\n-----\nFN|TN")
    #print ("TP|FP\n---\nFN|TN")
    for row in range(len(result)): #[[IS_ASSIGNED_OK?][REAL_GROUP][CHOSED_GROUP]];[[][][]];...
        if result[row][1]==iris_name and result[row][2]==iris_name:
            matrix[0][0]+=1 #tp_counter
        elif result[row][1]!=iris_name and result[row][2]==iris_name:
            matrix[0][1]+=1 #fp_counter
        elif result[row][1]==iris_name and result[row][2]!=iris_name:
            matrix[1][0]+=1 #fn_counter
        elif result[row][1]!=iris_name and result[row][2]!=iris_name:
            matrix[1][1]+=1 #tn_counter
    print("\n",matrix[0][0],"|",matrix[0][1],"\n-------\n",matrix[1][0],"|",matrix[1][1])

def perceptron_banknote_authentication(file_name, learning_rate, n_epoch,weights):

    data=list()
    data=loading_txt(file_name)
    #print(data)
    weights = train_weights(data, learning_rate, n_epoch,weights)





# test predictions
file_name="data_banknote_authentication.txt"
l_rate = 0.1
n_epoch = 20
weights=[0.0,0.0,0.0,0.0,0.0] #weights[0] is bias

perceptron_banknote_authentication(file_name, l_rate, n_epoch,weights)




