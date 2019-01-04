import landmarks as l2
import numpy as np

def get_data():
    X, y = l2.extract_features_labels()
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:4000]
    tr_Y = Y[:4000]
    te_X = X[4000:]
    te_Y = Y[4000:]
    return tr_X, tr_Y, te_X, te_Y




trainx, train_Y, testx, test_Y = get_data()
train_X = trainx.reshape(trainx.shape[0],trainx.shape[1]*trainx.shape[2])
test_X = testx.reshape(testx.shape[0],testx.shape[1]*testx.shape[2])
train_Y= train_Y[:,0]
test_Y= test_Y[:,0]

def SVM(trainX, trainY, testX):
    from sklearn import svm
    clf = svm.SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape='ovr', degree=2, gamma='scale', kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    
    clf.fit(trainX, trainY)
    return np.ravel(clf.predict(testX))

predict_Y = SVM(train_X,train_Y,test_X)
size = len(predict_Y)






def precisioncalculator(resultdata,testdata,sizedata):
    np.ravel(testdata)
    counter = 0
    for x in range (0, sizedata - 1):
        if resultdata[x] == testdata[x]:
            counter = counter + 1
    return counter/sizedata

precision = precisioncalculator(predict_Y, test_Y, size)*100
print(precision)
