from sklearn.neighbors import KNeighborsClassifier
import data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report,accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import pickle
import numpy as np
from sklearn.model_selection import cross_validate

X_undersample, y_undersample= data.undersample()
X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=1) # 70% training and 30% test

## gridsearch to find best tree possible
param_grid = {
        'n_estimators': [25, 50, 70,100], 
        'learning_rate':[0.1, 0.2, 0.5, 1, 1.5]
}

def show_result(clf):
    means_acc = clf.cv_results_['mean_test_accuracy']
    means_auc = clf.cv_results_['mean_test_roc_auc']
#    stds = clf.cv_results_['std_test_score']
    for acc, auc, params in zip(means_acc, means_auc, clf.cv_results_['params']):
        print("ACC: %0.3f AUC: %0.3f for %r"
              % (acc, auc , params))    

Pkl_Filename='dt_model.sav'
with open(Pkl_Filename, 'rb') as file:  
    model = pickle.load(file)
        
abc = AdaBoostClassifier(base_estimator=model)
    
clf = GridSearchCV(estimator=abc, param_grid=param_grid, scoring=['accuracy','roc_auc'],refit='accuracy', cv=5, n_jobs=4)
clf.fit(X_undersample,y_undersample)
print('BEST Parametesr : ', clf.best_params_ )

index= clf.best_index_ 
print('BEST accuracy :{}, best AUC: {}'.format(clf.cv_results_['mean_test_accuracy'][index],clf.cv_results_['mean_test_roc_auc'][index]))

def plot_gridsearch(clf,param_grid):
    
    score_mean = clf.cv_results_['mean_test_accuracy'].reshape(len(param_grid['n_estimators']),len(param_grid['learning_rate']))
    
#    plt.bar(KERNEL, score)
    for index, k in enumerate(param_grid['n_estimators']):
        plt.plot(param_grid['learning_rate'], score_mean[index,], '-o', label='n_estimators-'+str(k))
        
#    plt.legend()
    plt.title("Grid Search Scores",  fontweight='bold')
        
    plt.xlabel('learning_rate')
    plt.ylabel('Mean Test accuracy')
    plt.legend()
#    plt.show()
    plt.savefig('boosting-gridsearch.png')

show_result(clf)
plot_gridsearch(clf,param_grid)

ccp_alpha = [0.001, 0.0015, 0.002, 0.0025, 0.003]

##
acc_dict = {}
print(' apply more aggresive tuning here:' )
for c in ccp_alpha:
    model.ccp_alpha=c
    abc_ = AdaBoostClassifier(base_estimator= model, n_estimators=70, learning_rate=0.2)
    abc_.fit(X_train,y_train)
    y_pred = abc_.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    acc_dict[c] = acc
    print('ccp_alpha: {}, accuracy:{}'.format(c, acc))

plt.figure() 
plt.plot(acc_dict.keys(), acc_dict.values(), '-o') 
plt.xticks(np.arange(0.001, 0.0035, .0005))
plt.title("accuracy over ccp_alpha",  fontweight='bold')
    
plt.xlabel('ccp_alpha')
plt.ylabel('Mean Test accuracy')
#    plt.show()
plt.savefig('boosting-bar.png')

model.ccp_alpha= 0.002
best_dt_classifier = model
best_dt_classifier.fit(X_train, y_train)
print('best_dt_classifier at ccp_alpha: 0.002, Number of nodes {}, max_depth {}'.format(best_dt_classifier.tree_.node_count, best_dt_classifier.tree_.max_depth))

best_abc = AdaBoostClassifier(best_dt_classifier, n_estimators=70, learning_rate=0.2)
            
scores = cross_validate(best_abc, X_undersample, y_undersample, cv=5, scoring=['accuracy','roc_auc'], return_train_score=1)
print('Cross validation scores: average Train accuracy: ', np.mean(scores['train_accuracy']))
print('Cross validation scores: average Train AUC: ', np.mean(scores['train_roc_auc']))
print('Cross validation scores: average Test accuracy: ', np.mean(scores['test_accuracy']))
print('Cross validation scores: average Test AUC: ', np.mean(scores['test_roc_auc']))
