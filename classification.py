import landmarks as l2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import csv


def get_features():
    X=l2.extract_features_labels()
    return X

X,image_numbers, num=get_features()

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
     #Y = np.array([y, -(y - 1)]).T
     
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

        
def classify_eyeglasses(test_samples):
    print_arrray=[]
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(2,test_samples)
    #Y_train=Y_train[:,0]
    #Y_test=Y_test[:,0]
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
    #sprint_array=[np.array(image_test).T, np.array(Y_pred_te).T, np.array(Y_test).T]]

    with open('task_3.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        Y_test_column=np.array(Y_test).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i]])
    



def classify_emotion(test_samples):
    print_arrray=[]
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(3,test_samples)
    #Y_train=Y_train[:,0]
    #Y_test=Y_test[:,0]
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
    #sprint_array=[np.array(image_test).T, np.array(Y_pred_te).T, np.array(Y_test).T]]

    with open('task_1.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        Y_test_column=np.array(Y_test).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i]])


   
def classify_age(test_samples):
    print_arrray=[]
    X_train, X_test=divide_x(test_samples)
    image_train, image_test= divide_img(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(4,test_samples)
    #Y_train=Y_train[:,0]
    #Y_test=Y_test[:,0]
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
    #sprint_array=[np.array(image_test).T, np.array(Y_pred_te).T, np.array(Y_test).T]]

    with open('task_2.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        Y_test_column=np.array(Y_test).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i]])


    
    
    
number_of_train_samples= int(70/100*num)
number_of_test_samples= int(30/100*num)

def classify_human(test_samples):
    print_arrray=[]   
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
    #sprint_array=[np.array(image_test).T, np.array(Y_pred_te).T, np.array(Y_test).T]]

    with open('task_4.csv', mode='w') as output_file:
        results_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        image_column= np.array(image_test).T
        Y_pred_column=np.array(Y_pred_te).T
        Y_test_column=np.array(Y_test).T
        results_writer.writerow([ print_acc])
        for i in range(0, number_of_test_samples-1):
            results_writer.writerow([image_column[i], Y_pred_column[i]])
            

        
        




classify_eyeglasses(number_of_train_samples)
classify_emotion(number_of_train_samples)
classify_age(number_of_train_samples)
classify_human(number_of_train_samples)

