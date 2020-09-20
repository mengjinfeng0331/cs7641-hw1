from sklearn.neighbors import KNeighborsClassifier
import data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report,accuracy_score
from sklearn.model_selection import cross_validate
import numpy as np

X_undersample, y_undersample= data.undersample()
X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=1) # 70% training and 30% test

knn = KNeighborsClassifier()
print('KNN classifer')

## gridsearch to find best tree possible
param_grid = {
        'n_neighbors': [1,3,5,7,9,11], 
        'metric': ['euclidean', 'manhattan'],
        'weights' : [ 'uniform', 'distance']
}

def plot_gridsearch(clf,param_grid):
    
    score_mean = clf.cv_results_['mean_test_score']
    
    plt.plot(param_grid['n_neighbors'], score_mean, '-o')
        
#    plt.legend()
    plt.title("Grid Search Scores",  fontweight='bold')
        
    plt.xlabel('K')
    plt.ylabel('Mean Test score')
#    plt.legend()
#    plt.show()
    plt.savefig('knn-gridsearch.png')

def show_result(clf):
    means_acc = clf.cv_results_['mean_test_accuracy']
    means_auc = clf.cv_results_['mean_test_roc_auc']
#    stds = clf.cv_results_['std_test_score']
    for acc, auc, params in zip(means_acc, means_auc, clf.cv_results_['params']):
        print("ACC: %0.3f AUC: %0.3f for %r"
              % (acc, auc , params))    
        

clf = GridSearchCV(estimator=knn, param_grid=param_grid, scoring=['accuracy','roc_auc'],refit='accuracy', cv=5, n_jobs=4)
clf.fit(X_undersample,y_undersample)

best_model =clf.best_estimator_
print('BEST Parametesr : ', clf.best_params_ )
#
index= clf.best_index_ 
print('BEST accuracy :{}, best AUC: {}'.format(clf.cv_results_['mean_test_accuracy'][index],clf.cv_results_['mean_test_roc_auc'][index]))

### final model
scores = cross_validate(best_model, X_undersample, y_undersample, cv=5, scoring=['accuracy','roc_auc'], return_train_score=1)
print('Cross validation scores: average Train accuracy: ', np.mean(scores['train_accuracy']))
print('Cross validation scores: average Train AUC: ', np.mean(scores['train_roc_auc']))
print('Cross validation scores: average Test accuracy: ', np.mean(scores['test_accuracy']))
print('Cross validation scores: average Test AUC: ', np.mean(scores['test_roc_auc']))