"""
KNN and SVM learning.
Author: Yongxin (Fiona) Xu
Date: 12/19/2019
"""
# from python libraries
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np

# from my own libraries
import util

def train(learner, train_X, train_y, val_X, val_y):
    '''
    Train a learner and return training and testing accuracy,
    used for finding the best parameters
    Args:
        learner: an sklearn classifier
        train_X(np.ndarray): training data's features
        train_y(np.ndarray): training data's labels
        val_X(np.ndarray): validation data's features
        val_y(np.ndarray): validation data's labels
    Returns:
        (float): training score
        (float): validation score
    '''
    
    learner.fit(train_X, train_y)
    return learner.score(train_X, train_y), learner.score(val_X, val_y)


def test(learner, train_X, train_y, val_X, val_y, filename):
    """
    Test a leaner, return training and testing accuracy and confusion matrix,
    used for learning the performance of the model with the best parameters
    Args:
        learner: an sklearn classifier
        train_X(np.ndarray): training data's features
        train_y(np.ndarray): training data's labels
        val_X(np.ndarray): validation data's features
        val_y(np.ndarray): validation data's labels
    """
    
    learner.fit(train_X, train_y)
    y_pred = learner.predict(val_X)
    print('Training accuracy: ', learner.score(train_X, train_y))
    print('Testing accuracy: ', learner.score(val_X, val_y))
    cm = confusion_matrix(val_y, y_pred)
    print('Confusion matrix:\n', cm)
    util.draw_heatmap(cm, filename)
 
            
def main(plot):
    train_X, train_y, val_X, val_y = util.load_data('./data/semeion.data', 0.8)
    my_X, my_y = util.get_my_data('./data/testset.csv', './data/digits/')

    # KNN
    print('------KNN------')
    if plot:
        # find the best hyper-parameters
        # initialize lists for plotting graph
        uni_train_accur, uni_test_accur, dist_train_accur, dist_test_accur = ([] for _ in range(4))
        n_neighbors = np.arange(1,11)
        for neighbors in n_neighbors:
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
    test(knn_clf, train_X, train_y, val_X, val_y, './img/knn_cm.png')
 
    # find the wrong predictions
    if plot:
        util.draw_wrong_img(val_X, val_y, y_pred, 'knn')
    
    my_y_pred = knn_clf.predict(my_X)
    print('Testing accuracy of our own data: ', knn_clf.score(my_X, my_y))
    cm = confusion_matrix(my_y, my_y_pred)
    print('Confusion matrix of our own data:\n', cm)
    util.draw_heatmap(cm, './img/knn_own.png')

    if plot:
        util.draw_wrong_img(my_X, my_y, my_y_pred, 'knn_own')
        
    
    # SVM
    print('------SVM------')
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
    test(svm_clf, train_X, train_y, val_X, val_y, './img/svm_cm.png')
    
    # find the wrong predictions
    if plot:
        util.draw_wrong_img(val_X, val_y, y_pred, 'svm')
    
    my_y_pred = svm_clf.predict(my_X)
    print('Testing accuracy of our own data: ', svm_clf.score(my_X, my_y))
    cm =  confusion_matrix(my_y, my_y_pred)
    print('Confusion matrix of our own data:\n', cm)
    util.draw_heatmap(cm, './img/svm_own.png')
        
    if plot:
        util.draw_wrong_img(my_X, my_y, my_y_pred, 'svm_own')
        
if __name__ == '__main__':
    main(False)