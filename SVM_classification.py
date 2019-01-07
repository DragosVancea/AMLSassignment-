import SVM_preprocessing as l2
import CNN_preprocessing as prep
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import csv
import tensorflow as tf
from tensorflow import keras
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D



#receive face features from SVM_preprocessing.py
def get_features():
    X,image_numbers, num = l2.extract_features_labels()
    return X, image_numbers, num


X,image_numbers, num=get_features()
number_of_train_samples= int(70/100*num)
number_of_test_samples= int(30/100*num)

#divides independent variables into training and testing data
def divide_x(test_samples):

    tr_X = X[:test_samples]
    te_X = X[test_samples:]
    return tr_X, te_X

#divides the corresponding image names into training and testing data
def divide_img(test_samples):

    tr_img = image_numbers[:test_samples]
    te_img = image_numbers[test_samples:]
    return tr_img, te_img



#receives labels from SVM_preprocessing.py and divides them into training and testing data
def get_labels(i,test_samples):
    
     Y=l2.extract_labels(i)
     tr_Y = Y[:test_samples]
     te_Y = Y[test_samples:]
     
     
     return tr_Y, te_Y
    

def SVM_eyeglasses(x_tr, y_tr, x_te):
    #Create the model
    model = svm.SVC(C=1,  degree=2, gamma='scale', kernel='poly')
    #Fit the model
    model.fit(x_tr, y_tr)
    return np.ravel(model.predict(x_te))

def SVM_emotion(x_tr, y_tr, x_te):
    #Create the model
    model = svm.SVC(C=5,  degree=2, gamma='scale', kernel='poly')
    #Fit the model
    model.fit(x_tr, y_tr)
    return np.ravel(model.predict(x_te))

def SVM_age(x_tr, y_tr, x_te):
    #Create the model
    model = svm.SVC(C=5, gamma='scale', kernel='rbf')
    #Fit the model
    model.fit(x_tr, y_tr)
    return np.ravel(model.predict(x_te))

def SVM_human(x_tr, y_tr, x_te):
    #Create the model
    model = svm.SVC(C=5,  degree=2, gamma='scale', kernel='poly')
    #Fit the model
    model.fit(x_tr, y_tr)
    return np.ravel(model.predict(x_te))

        
def classify_eyeglasses_with_SVM(test_samples):
    #Call division the functions
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)

    #Reshapes x to a two dimensional array
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])

    
    Y_train, Y_test=get_labels(2,test_samples)

    #Does prediction based on the fitted model
    Y_pred_te = SVM_eyeglasses(X_train,Y_train,X_test)    
    size_te = len(Y_pred_te)
    
    #Y_pred_tr = SVM_eyeglasses(X_train,Y_train,X_train)
    #size_tr = len(Y_pred_tr)
    
    #print("Accuracy on test data for glasses detection:")
    #print(accuracy_score(Y_test, Y_pred_te)*100)
    #print("Accuracy on train data for glasses detection:")
    #print(accuracy_score(Y_train, Y_pred_tr)*100)
    #print("The confusion matrix is:")
    #print(confusion_matrix(Y_test, Y_pred_te))

    

    print_acc=accuracy_score(Y_test, Y_pred_te)
    
    #Print the predictions
    with open('task_3.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])
    



def classify_emotion_with_SVM(test_samples):
    #Call division the functions
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)


    #Reshapes x to a two dimensional array
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(3,test_samples)

    #Does prediction based on the fitted model
    Y_pred_te = SVM_emotion(X_train,Y_train,X_test)
    size_te = len(Y_pred_te)
    #Y_pred_tr = SVM_emotion(X_train,Y_train,X_train)
    #size_tr = len(Y_pred_tr)
    #print("Accuracy on test data for emotion detection:")
    #print(accuracy_score(Y_test, Y_pred_te)*100)
    #print("Accuracy on train data for emotion detection:")
    #print(accuracy_score(Y_train, Y_pred_tr)*100)
    #print("The confusion matrix is:")
    #print(confusion_matrix(Y_test, Y_pred_te))



    print_acc=accuracy_score(Y_test, Y_pred_te)
    
    #Print the predictions
    with open('task_1.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])


   
def classify_age_with_SVM(test_samples):
    #Call division the functions
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)

    #Reshapes x to a two dimensional array
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(4,test_samples)
    #Does prediction based on the fitted model
    Y_pred_te = SVM_age(X_train,Y_train,X_test)
    size_te = len(Y_pred_te)
    
    #Y_pred_tr = SVM_age(X_train,Y_train,X_train)
    #size_tr = len(Y_pred_tr)
    #print("Accuracy on test data for age detection:")
    #print(accuracy_score(Y_test, Y_pred_te)*100)
    #print("Accuracy on train data for age detection:")
    #print(accuracy_score(Y_train, Y_pred_tr)*100)
    #print("The confusion matrix is:")
    #print(confusion_matrix(Y_test, Y_pred_te))


    print_acc=accuracy_score(Y_test, Y_pred_te)
    
    #Print the predictions
    with open('task_2.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])



def classify_human_with_SVM(test_samples):
    #Call division the functions
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)

    #Reshapes x to a two dimensional array
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])

    
    Y_train, Y_test=get_labels(5,test_samples)

    #Does prediction based on the fitted model
    Y_pred_te = SVM_human(X_train,Y_train,X_test)
    size_te = len(Y_pred_te)
    #Y_pred_tr = SVM_human(X_train,Y_train,X_train)
    #size_tr = len(Y_pred_tr)
    #print("Accuracy on test data for human detection:")
    #print(accuracy_score(Y_test, Y_pred_te)*100)
    #print("Accuracy on train data for human detection:")
    #print(accuracy_score(Y_train, Y_pred_tr)*100)
    #print("The confusion matrix is:")
    #print(confusion_matrix(Y_test, Y_pred_te))

    
    print_acc=accuracy_score(Y_test, Y_pred_te)
    #Print the predictions
    with open('task_4.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])

#These functions are called from the main class
            
def SVM_eyeglasses_classification():   
    classify_eyeglasses_with_SVM(number_of_train_samples)
    
def SVM_emotion_classification():
    classify_emotion_with_SVM(number_of_train_samples)
    
def SVM_age_classification():
    classify_age_with_SVM(number_of_train_samples)
    
def SVM_human_classification():
    classify_human_with_SVM(number_of_train_samples)

