import landmarks as l2
import preprocessing as prep
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




def get_features():
    X=l2.extract_features_labels()
    return X

#X,image_numbers, num=get_features()
#number_of_train_samples= int(70/100*num)
#number_of_test_samples= int(30/100*num)

def divide_x(test_samples):

    tr_X = X[:test_samples]
    te_X = X[test_samples:]
    return tr_X, te_X


def divide_img(test_samples):

    tr_img = image_numbers[:test_samples]
    te_img = image_numbers[test_samples:]
    return tr_img, te_img




def get_labels(i,test_samples):
    
     Y=l2.extract_labels(i)
     print(Y)
     tr_Y = Y[:test_samples]
     te_Y = Y[test_samples:]
     
     
     return tr_Y, te_Y

def SVM_eyeglasses(x_tr, y_tr, x_te):
    
    clf = svm.SVC(C=1,  degree=2, gamma='scale', kernel='poly')
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))

def SVM_emotion(x_tr, y_tr, x_te):
    
    clf = svm.SVC(C=1,  degree=2, gamma='scale', kernel='poly')
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))

def SVM_age(x_tr, y_tr, x_te):
    
    clf = svm.SVC(C=10,  degree=2, gamma='scale', kernel='rbf')
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))

def SVM_human(x_tr, y_tr, x_te):
    
    clf = svm.SVC(C=1,  degree=2, gamma='scale', kernel='poly')
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))

        
def classify_eyeglasses_with_SVM(test_samples):
    
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(2,test_samples)
    
    Y_pred_te = SVM_eyeglasses(X_train,Y_train,X_test)    
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_eyeglasses(X_train,Y_train,X_train)
    size_tr = len(Y_pred_tr)
    print("Accuracy on test data for glasses detection:")
    print(accuracy_score(Y_test, Y_pred_te)*100)
    print("Accuracy on train data for glasses detection:")
    print(accuracy_score(Y_train, Y_pred_tr)*100)
    print("The confusion matrix is:")
    print(confusion_matrix(Y_test, Y_pred_te))

    

    print_acc=accuracy_score(Y_test, Y_pred_te)
    

    with open('task_3.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i]])
    



def classify_emotion_with_SVM(test_samples):
   
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(3,test_samples)
    Y_pred_te = SVM_emotion(X_train,Y_train,X_test)
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_emotion(X_train,Y_train,X_train)
    size_tr = len(Y_pred_tr)
    print("Accuracy on test data for emotion detection:")
    print(accuracy_score(Y_test, Y_pred_te)*100)
    print("Accuracy on train data for emotion detection:")
    print(accuracy_score(Y_train, Y_pred_tr)*100)
    print("The confusion matrix is:")
    print(confusion_matrix(Y_test, Y_pred_te))



    print_acc=accuracy_score(Y_test, Y_pred_te)
    

    with open('task_1.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i]])


   
def classify_age_with_SVM(test_samples):
    
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(4,test_samples)
    Y_pred_te = SVM_age(X_train,Y_train,X_test)
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_age(X_train,Y_train,X_train)
    size_tr = len(Y_pred_tr)
    print("Accuracy on test data for age detection:")
    print(accuracy_score(Y_test, Y_pred_te)*100)
    print("Accuracy on train data for age detection:")
    print(accuracy_score(Y_train, Y_pred_tr)*100)
    print("The confusion matrix is:")
    print(confusion_matrix(Y_test, Y_pred_te))


    print_acc=accuracy_score(Y_test, Y_pred_te)
    

    with open('task_2.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i]])



def classify_human_with_SVM(test_samples):
   
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(5,test_samples)
    #Y_train=Y_train[:,0]
    #Y_test=Y_test[:,0]
    Y_pred_te = SVM_human(X_train,Y_train,X_test)
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_human(X_train,Y_train,X_train)
    size_tr = len(Y_pred_tr)
    print("Accuracy on test data for human detection:")
    print(accuracy_score(Y_test, Y_pred_te)*100)
    print("Accuracy on train data for human detection:")
    print(accuracy_score(Y_train, Y_pred_tr)*100)
    print("The confusion matrix is:")
    print(confusion_matrix(Y_test, Y_pred_te))

    
    print_acc=accuracy_score(Y_test, Y_pred_te)
    
    with open('task_4.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i]])
            




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
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    
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
    

    
    with open('task_3_CNN.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        Y_test_column=np.array(Y_test2).T
        print(image_column.shape)
        print( Y_test_column.shape)
        print( Y_pred_column.shape)
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i], Y_test_column[i]])




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
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    
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
    

    
    with open('task_1_CNN.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        Y_test_column=np.array(Y_test2).T
        print(image_column.shape)
        print( Y_test_column.shape)
        print( Y_pred_column.shape)
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i], Y_test_column[i]])






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
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    
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
    

    
    with open('task_2_CNN.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        Y_test_column=np.array(Y_test2).T
        print(image_column.shape)
        print( Y_test_column.shape)
        print( Y_pred_column.shape)
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i], Y_test_column[i]])




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
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    
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
    

    
    with open('task_4_CNN.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        Y_test_column=np.array(Y_test2).T
        print(image_column.shape)
        print( Y_test_column.shape)
        print( Y_pred_column.shape)
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i], Y_test_column[i]])





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
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    
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
    

    
    with open('task_5.csv', mode='w') as output_file:
        results_writer= csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(img_test).T
        Y_pred_column=np.array(Y_pred).T
        Y_test_column=np.array(Y_test2).T
        print(image_column.shape)
        print( Y_test_column.shape)
        print( Y_pred_column.shape)
        
        results_writer.writerow([ score[1]])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i], Y_test_column[i]])
    
    

 

     


#classify_eyeglasses_with_SVM(number_of_train_samples)
#classify_emotion_with_SVM(number_of_train_samples)
#classify_age_with_SVM(number_of_train_samples)
#classify_human_with_SVM(number_of_train_samples)


#classify_eyeglasses_with_CNN()
#classify_emotion_with_CNN()
#classify_age_with_CNN()
#classify_human_with_CNN()
#classify_hair()
