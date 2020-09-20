from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import data
from sklearn.model_selection import GridSearchCV
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pickle
import numpy as np
from sklearn.model_selection import cross_validate
import pandas as pd

X_undersample, y_undersample= data.undersample()
X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=1) # 70% training and 30% test

## initial test with default parameters:
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
print('Default Tree, Number of nodes {}, max_depth {}'.format(dt.tree_.node_count, dt.tree_.max_depth))
y_pred = dt.predict(X_train)
train_acc = accuracy_score(y_train, y_pred)
print('training accuracy : ', train_acc)
y_pred = dt.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print('test accuracy : ', test_acc)
print('\n'*2)

## pruning 
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
df = pd.DataFrame()
for index, ccp_alpha in enumerate(ccp_alphas):
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    df.loc[index, 'ccp_alpha'] = ccp_alpha
    df.loc[index, 'nodes'] = clf.tree_.node_count
    df.loc[index, 'depth'] = clf.tree_.max_depth
    print("ccp_alpha: {}, Nodes: {}, depth :{} ".format(round(ccp_alpha,5), clf.tree_.node_count,clf.tree_.max_depth ))
df[['nodes','depth']] = df[['nodes','depth']].astype(int)

def plot_alphas(clfs, ccp_alphas):
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("ccp_alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs ccp_alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("ccp_alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs ccp_alpha")
    fig.tight_layout()
    fig.savefig('dt_alphas.png')
    fig.show()

plot_alphas(clfs, ccp_alphas)

## gridsearch to find best tree possible
param_grid = {
        'criterion': ['gini','entropy'], 
        'ccp_alpha':[0.001, 0.0015, 0.002, 0.0025]
}

## 
print('Gridsearch for hyperParamters:')
dt = DecisionTreeClassifier()
clf = GridSearchCV(estimator=dt, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=4)
clf.fit(X_undersample,y_undersample)

def plot_gridsearch(clf,param_grid):
    
    score_mean = clf.cv_results_['mean_test_score'].reshape(4,2)
    plt.figure()
    
#    plt.bar(KERNEL, score)
    for index, k in enumerate(param_grid['criterion']):
        plt.plot(param_grid['ccp_alpha'], score_mean[:,index], '-o', label='criterion-'+k)
        
#    plt.legend()
    plt.title("Grid Search Scores",  fontweight='bold')
        
    plt.xlabel('ccp_alpha')
    plt.ylabel('Mean Test score')
    plt.legend()
#    plt.show()
    plt.savefig('dt-grid.png')

plot_gridsearch(clf,param_grid)   

#############################################################################
## post best model analysis
best_model =clf.best_estimator_
disp = plot_confusion_matrix(best_model, X_test, y_test,cmap=plt.cm.Blues,normalize='true')
disp.figure_.savefig('cm_dt.png')

### final model
scores = cross_validate(best_model, X_undersample, y_undersample, cv=5, scoring=['accuracy','roc_auc'], return_train_score=1)
print('Cross validation scores: average Train accuracy: ', np.mean(scores['train_accuracy']))
print('Cross validation scores: average Train AUC: ', np.mean(scores['train_roc_auc']))
print('Cross validation scores: average Test accuracy: ', np.mean(scores['test_accuracy']))
print('Cross validation scores: average Test AUC: ', np.mean(scores['test_roc_auc']))

## save model
filename = 'dt_model.sav'
pickle.dump(best_model, open(filename, 'wb'))
