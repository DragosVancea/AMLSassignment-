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




#CNN classification for the eyeglasses
def classify_eyeglasses_with_CNN():


    #data received from the CNN_preprocessing file
    num, X_train, X_test, Y_train, Y_test, img_train, img_test =  prep.division(2)

    #Converting the data to numpy arrays
    X_train=np.array(X_train)
    X_test=np.array(X_test)    
    img_train=np.array(img_train)
    img_test=np.array(img_test)

    # two classes 0 and 1
    num_classes= 2
    Y_test2=Y_test

    
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    batch_size = 16
    epochs = 5

    #Implememting the CNN
    # This line of code allowed the creation of a sequential model, as the convolutional neural network usually is.
    model = Sequential()
    #This line of code added to the model the first convolutional layer.
    model.add(Conv2D(32, kernel_size=(9,9), activation='relu', input_shape=(256,256,3)))
    #Pooling Layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Another Convolutional Layer
    model.add(Conv2D(32, (9,9), activation='relu'))
    #Dropout Layer
    model.add(Dropout(0.25))
    #Flattens the input
    model.add(Flatten())
    #Dense layer. All neurons connected.
    model.add(Dense(128, activation='relu'))
    #Dropout Layer
    model.add(Dropout(0.5))
    #Dense layer. All neurons connected.
    model.add(Dense(num_classes, activation='softmax'))
    #Compiling the model.
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    #Fiting the model based on the data
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(X_test, Y_test))
    #Evaluating performance
    score = model.evaluate(X_test, Y_test, verbose=0)
    
    #Predicting based on test data
    result=model.predict(X_test)
    Y_pred=[]
    i=0

    
    number_of_test_samples=int(0.3*num)

    #Assigns the Y_pred the maximum probability from the output layer
    for res in result:
        max1=0
        c=0
        j=0
        for resj in result[i]:
            if(result[i,j]>max1):
                max1=result[i,j]
                c=j
            j=j+1
            
        Y_pred.append(c)
        i=i+1
    

    #Prints the accuracy and the predictions to a .csv file and makes 0 class back to -1
    with open('task_3.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])



#CNN classification for the emotion
def classify_emotion_with_CNN():


    #data received from the CNN_preprocessing file
    num, X_train, X_test, Y_train, Y_test, img_train, img_test =  prep.division(3)


    #Converting the data to numpy arrays
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    img_train=np.array(img_train)
    img_test=np.array(img_test)

    # two classes 0 and 1
    num_classes= 2
    Y_test2=Y_test
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    batch_size = 16
    epochs = 5
    #Implememting the CNN
    # This line of code allowed the creation of a sequential model, as the convolutional neural network usually is.
    model = Sequential()
    #This line of code added to the model the first convolutional layer.
    model.add(Conv2D(32, kernel_size=(9,9), activation='relu', input_shape=(256,256,3)))
    #Pooling Layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Another Convolutional Layer
    model.add(Conv2D(32, (9,9), activation='relu'))
    #Dropout Layer
    model.add(Dropout(0.5))
    #Flattens the input
    model.add(Flatten())
    #Dense layer. All neurons connected.
    model.add(Dense(128, activation='relu'))
    #Dropout Layer
    model.add(Dropout(0.75))
    #Dense layer. All neurons connected.
    model.add(Dense(num_classes, activation='softmax'))

    #Compiling the model.
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    #Fiting the model based on the data
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
    #Evaluating performance
    score = model.evaluate(X_test, Y_test)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])

    
    #Predicting based on test data
    result=model.predict(X_test)
    Y_pred=[]
    i=0

    
    number_of_test_samples=int(0.3*num)

    #Assigns the Y_pred the maximum probability from the output layer
    for res in result:
        max1=0
        c=0
        j=0
        for resj in result[i]:
            if(result[i,j]>max1):
                max1=result[i,j]
                c=j
            j=j+1
            
        Y_pred.append(c)
        i=i+1
    

    #Prints the accuracy and the predictions to a .csv file and makes 0 class back to -1
    with open('task_1.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])




#CNN classification for the age

def classify_age_with_CNN():


    #data received from the CNN_preprocessing file
    num, X_train, X_test, Y_train, Y_test, img_train, img_test =  prep.division(4)


    #Converting the data to numpy arrays
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    img_train=np.array(img_train)
    img_test=np.array(img_test)
    num_classes= 2
    Y_test2=Y_test
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    batch_size = 16
    epochs = 5
    #CNN implementation
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9,9), activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (9,9), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(num_classes, activation='softmax'))
    #CNN compilation and model fit
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    #CNN prediction
    result=model.predict(X_test)
    Y_pred=[]
    i=0

    
    number_of_test_samples=int(0.3*num)

    #Assigns the Y_pred the maximum probability from the output layer
    for res in result:
        max1=0
        c=0
        j=0
        for resj in result[i]:
            if(result[i,j]>max1):
                max1=result[i,j]
                c=j
            j=j+1
            
        Y_pred.append(c)
        i=i+1
    

    #Prints the accuracy and the predictions to a .csv file and makes 0 class back to -1
    with open('task_2.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])


#CNN classification for the humnan\nothuman

def classify_human_with_CNN():


    #data received from the CNN_preprocessing file
    num, X_train, X_test, Y_train, Y_test, img_train, img_test =  prep.division(5)

    #Converting the data to numpy arrays
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    img_train=np.array(img_train)
    img_test=np.array(img_test)
    num_classes= 2
    Y_test2=Y_test
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    batch_size = 16
    epochs = 5
    #CNN implementation
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9,9), activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (9,9), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(num_classes, activation='softmax'))
    #CNN compilation and model fit
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    

    #CNN prediction
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result=model.predict(X_test)
    Y_pred=[]
    i=0

    
    number_of_test_samples=int(0.3*num)

    #Assigns the Y_pred the maximum probability from the output layer
    for res in result:
        max1=0
        c=0
        j=0
        for resj in result[i]:
            if(result[i,j]>max1):
                max1=result[i,j]
                c=j
            j=j+1
            
        Y_pred.append(c)
        i=i+1
    

    #Prints the accuracy and the predictions to a .csv file and makes 0 class back to -1
    with open('task_4.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T

        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])




#CNN classification for the hair colour
def classify_hair():


    #data received from the CNN_preprocessing file
    num, X_train, X_test, Y_train, Y_test, img_train, img_test =  prep.division(1)

    #Converting the data to numpy arrays
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    img_train=np.array(img_train)
    img_test=np.array(img_test)
    num_classes= 6
    Y_test2=Y_test
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    batch_size = 16
    epochs = 15
    #CNN implementation
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9,9), activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (9,9), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(num_classes, activation='softmax'))
    #CNN compilation and model fit
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    

    #CNN prediction
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result=model.predict(X_test)
    Y_pred=[]
    i=0

    
    number_of_test_samples=int(0.3*num)

    #Assigns the Y_pred the maximum probability from the output layer
    for res in result:
        max1=0
        c=0
        j=0
        for resj in result[i]:
            if(result[i,j]>max1):
                max1=result[i,j]
                c=j
            j=j+1
            
        Y_pred.append(c)
        i=i+1
    
    
    #Prints the accuracy and the predictions to a .csv file
    with open('task_5.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        Y_test_column=np.array(Y_test2).T
        
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i]])


#Function called from main.py

def CNN_eyeglasses():
    classify_eyeglasses_with_CNN()
    
def CNN_emotion():
    classify_emotion_with_CNN()

def CNN_age():
    classify_age_with_CNN()

def CNN_human():
    classify_human_with_CNN()

def CNN_hair():
    classify_hair()





