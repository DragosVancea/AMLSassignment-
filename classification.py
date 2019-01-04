import landmarks as l2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



def get_features():
    X=l2.extract_features_labels()
    return X

X=get_features()

def divide_x(test_samples):
    tr_X = X[:test_samples]
    te_X = X[test_samples:]

    #tr_X = X[:100]
    #te_X = X[100:]
    return tr_X, te_X






def get_labels(i,test_samples):
    
     y=l2.extract_labels(i)
     Y = np.array([y, -(y - 1)]).T
     
     tr_Y = Y[:test_samples]
     te_Y = Y[test_samples:]
     
     #tr_Y = Y[:100]
     #te_Y = Y[100:]
     
     return tr_Y, te_Y

def SVM_eyeglasses(x_tr, y_tr, x_te):
    
    clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=2, gamma='scale', kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False) 
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))

def SVM_emotion(x_tr, y_tr, x_te):
    
    clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=2, gamma='scale', kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False) 
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))

def SVM_age(x_tr, y_tr, x_te):
    
    clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=2, gamma='scale', kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False) 
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))

def SVM_human(x_tr, y_tr, x_te):
    
    clf = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=2, gamma='scale', kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False) 
    clf.fit(x_tr, y_tr)
    return np.ravel(clf.predict(x_te))






def accuracy_calc_te(y_pred,y_te,size):

    np.ravel(y_te)
    num = 0
    for i in range (0, size - 1):
        if y_pred[i] == y_te[i]:
            num = num + 1
    return num/size

def accuracy_calc_tr(y_pred,y_tr,size):

    np.ravel(y_tr)
    num = 0
    for i in range (0, size - 1):
        if y_pred[i] == y_tr[i]:
            num = num + 1
    return num/size




def classify_eyeglasses(test_samples):

    X_train, X_test=divide_x(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(2,test_samples)
    Y_train=Y_train[:,0]
    Y_test=Y_test[:,0]
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


def classify_emotion(test_samples):

    X_train, X_test=divide_x(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(3,test_samples)
    Y_train=Y_train[:,0]
    Y_test=Y_test[:,0]
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

def classify_age(test_samples):

    X_train, X_test=divide_x(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(4,test_samples)
    Y_train=Y_train[:,0]
    Y_test=Y_test[:,0]
    Y_pred_te = SVM_age(X_train,Y_train,X_test)
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_age(X_train,Y_train,X_train)
    size_tr = len(Y_pred_tr)
    #accuracy_te = accuracy_calc_te(Y_pred_te, Y_test, size_te)*100
    print("Accuracy on test data for age detection:")
    print(accuracy_score(Y_test, Y_pred_te)*100)
    #accuracy_tr = accuracy_calc_tr(Y_pred_tr, Y_train, size_tr)*100
    print("Accuracy on train data for age detection:")
    print(accuracy_score(Y_train, Y_pred_tr)*100)

    print("The confusion matrix is:")
    print(confusion_matrix(Y_test, Y_pred_te))

def classify_human(test_samples):
    
    X_train, X_test=divide_x(test_samples)
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    Y_train, Y_test=get_labels(4,test_samples)
    Y_train, Y_test=get_labels(5,test_samples)
    Y_train=Y_train[:,0]
    Y_test=Y_test[:,0]
    Y_pred_te = SVM_human(X_train,Y_train,X_test)
    size_te = len(Y_pred_te)
    Y_pred_tr = SVM_human(X_train,Y_train,X_train)
    size_tr = len(Y_pred_tr)
    accuracy_te = accuracy_calc_te(Y_pred_te, Y_test, size_te)*100
    print("Accuracy on test data for human detection:")
    print(accuracy_score(Y_test, Y_pred_te)*100)
    print("Accuracy on train data for human detection:")
    print(accuracy_score(Y_train, Y_pred_tr)*100)

    print("The confusion matrix is:")
    print(confusion_matrix(Y_test, Y_pred_te))


classify_eyeglasses(3000)
classify_emotion(3000)
classify_age(3000)
#classify_human(3000)

