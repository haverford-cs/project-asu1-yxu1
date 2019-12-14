# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import glob
import matplotlib.pyplot as plt
import numpy as np

# from my own libraries
import util

'''
def runTuneTest(learner, params, data):
    
    # divide the data into training/test 5 folds

    for _ in range(5):
        clf = GridSearchCV(learner, params, cv=3)
        # fit the classifier using your training data
        clf.fit(data['t_X'], data['t_y'])
        # get paramenters for this fold
        print(clf.best_params_)
        # get the train-set accuracy
        print("Training Score: ", clf.score(data['t_X'], data['t_y']))
        print("Testing Score: ", clf.score(data['v_X'], data['v_y']))
'''

def train(learner, train_X, train_y, val_X, val_y):
    '''
    Train a learner and return training and testing accuracy
    '''
    learner.fit(train_X, train_y)
    return learner.score(train_X, train_y), learner.score(val_X, val_y)


            


def main(plot):
    train_X, train_y, val_X, val_y = util.load_data('./data/semeion.data', 0.8)
    
    # data = dict(t_X=train_X, t_y=train_y, v_X=val_X, v_y=val_y)
    
    # KNN
    if plot:
        # find the best hyper-parameters
        # initialize lists for plotting graph
        uni_train_accur, uni_test_accur, dist_train_accur, dist_test_accur = ([] for _ in range(4))
        n_neighbors = np.arange(1,11)
        for neighbors in n_neighbors:
            print(neighbors)
            for weight in ['uniform', 'distance']:
                knn_clf = KNeighborsClassifier(neighbors, weights=weight)
                train_accur, test_accur = train(knn_clf, train_X, train_y, val_X, val_y)
                
                if weight == 'uniform':
                    uni_train_accur.append(train_accur)
                    uni_test_accur.append(test_accur)
                else:
                    dist_train_accur.append(train_accur)
                    dist_test_accur.append(test_accur)
        
        # plot graph
        plt.plot(n_neighbors, uni_train_accur, 'C0--', n_neighbors, uni_test_accur, 'C0', \
            n_neighbors, dist_train_accur, 'C1--', n_neighbors, dist_test_accur, 'C1')
        plt.legend(['Train (uniform)', 'Test (uniform)', 'Train (distance)', 'Test (distance)'],\
            bbox_to_anchor=(1.04,1), loc="upper left")
        plt.xlabel('Number of neighbors')
        plt.ylabel('Accuracy')
        plt.savefig('./img/knn_score.png', bbox_inches="tight")
        plt.show()
  
    # set weights as distance, and n_neighbors as 5
    knn_clf = KNeighborsClassifier(5, weights='distance')
    knn_clf.fit(train_X, train_y)
    y_pred = knn_clf.predict(val_X)
    print('------KNN------')
    print('Testing accuracy: ', knn_clf.score(val_X, val_y))
    print('Confusion matrix:\n', confusion_matrix(val_y, y_pred))
    
    # find the wrong predictions
    if plot:
        util.draw_wrong_img(val_X, val_y, y_pred, 'knn')
        
    
    X = util.test_self_data()
    y_pred = knn_clf.predict(X)
    print(y_pred)

    # SVM
    if plot:
        # find the best hyper-parameters
        # initialize lists for plotting graph
        train1, test1, train2, test2, train3, test3 = ([] for _ in range(6))
        gamma = np.array([0.0001, 0.001, 0.01, 0.1, 1.])
        for g in gamma:
            for C in [1., 10., 100.]:
                svm_clf = SVC(C=C, gamma=g)
                train_accur, test_accur = train(svm_clf, train_X, train_y, val_X, val_y)
                
                if C == 1.:
                    train1.append(train_accur)
                    test1.append(test_accur)
                elif C == 10.:
                    train2.append(train_accur)
                    test2.append(test_accur)
                else:
                    train3.append(train_accur)
                    test3.append(test_accur)
                    
        # plot graph
        gamma=np.log10(gamma) 
        plt.plot(gamma, train1, 'C0--', gamma, test1, 'C0', gamma, train2, 'C1--', \
            gamma, test2, 'C1', gamma, train3, 'C2--', gamma, test3, 'C2')
        plt.legend(['Train (C=1)', 'Test (C=1)', 'Train (C=10)', 'Test (C=10)', \
            'Train (C=100)', 'Test (C=100)'],\
            bbox_to_anchor=(1.04,1), loc="upper left")
        plt.xlabel('Gamma (log)')
        plt.ylabel('Accuracy')
        plt.savefig('./img/svm_score.png', bbox_inches="tight")
        plt.show()
  
    # set C as 100, and gamma as 0.01
    svm_clf = SVC(C=100, gamma=0.01)
    svm_clf.fit(train_X, train_y)
    y_pred = svm_clf.predict(val_X)
    print('------SVM------')
    print('Testing accuracy: ', svm_clf.score(val_X, val_y))
    print('Confusion matrix:\n', confusion_matrix(val_y, y_pred))
    
    # find the wrong predictions
    if plot:
        util.draw_wrong_img(val_X, val_y, y_pred, 'svm')
        
    X = util.test_self_data()
    y_pred = svm_clf.predict(X)
    print(y_pred)
        
if __name__ == '__main__':
    main(False)