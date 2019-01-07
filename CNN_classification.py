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





def classify_eyeglasses_with_CNN():



    num, X_train, X_test, Y_train, Y_test, img_train, img_test =  prep.division(2)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    img_train=np.array(img_train)
    img_test=np.array(img_test)
    num_classes= 2
    Y_test2=Y_test
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    batch_size = 16
    epochs = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9,9), activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (9,9), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result=model.predict(X_test)
    Y_pred=[]
    i=0

    
    number_of_test_samples=int(0.3*num)

    
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
    

    
    with open('task_3.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])




def classify_emotion_with_CNN():



    num, X_train, X_test, Y_train, Y_test, img_train, img_test =  prep.division(3)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    img_train=np.array(img_train)
    img_test=np.array(img_test)
    num_classes= 2
    Y_test2=Y_test
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    batch_size = 16
    epochs = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9,9), activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (9,9), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

 
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result=model.predict(X_test)
    Y_pred=[]
    i=0

    
    number_of_test_samples=int(0.3*num)

    
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
    

    
    with open('task_1.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])






def classify_age_with_CNN():



    num, X_train, X_test, Y_train, Y_test, img_train, img_test =  prep.division(4)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    img_train=np.array(img_train)
    img_test=np.array(img_test)
    num_classes= 2
    Y_test2=Y_test
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    batch_size = 16
    epochs = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9,9), activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (9,9), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
   
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result=model.predict(X_test)
    Y_pred=[]
    i=0

    
    number_of_test_samples=int(0.3*num)

    
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
    

    
    with open('task_2.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])




def classify_human_with_CNN():



    num, X_train, X_test, Y_train, Y_test, img_train, img_test =  prep.division(5)
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    img_train=np.array(img_train)
    img_test=np.array(img_test)
    num_classes= 2
    Y_test2=Y_test
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    batch_size = 16
    epochs = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(9,9), activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (9,9), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    

    
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,  verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result=model.predict(X_test)
    Y_pred=[]
    i=0

    
    number_of_test_samples=int(0.3*num)

    
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
    

    
    with open('task_4.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T

        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            if Y_pred_column[i] == 0:
                Y_pred_column[i]= -1
            results_writer.writerow([image_column[i], Y_pred_column[i]])





def classify_hair():



    num, X_train, X_test, Y_train, Y_test, img_train, img_test =  prep.division(1)
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

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    

    
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    result=model.predict(X_test)
    Y_pred=[]
    i=0

    
    number_of_test_samples=int(0.3*num)

    
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
    
    
    
    with open('task_5.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        Y_test_column=np.array(Y_test2).T
        
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i]])


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





