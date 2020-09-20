# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
import data
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import backend as K
import time
import pandas as pd

df = pd.DataFrame()
X_undersample, y_undersample= data.undersample()
X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=1) # 70% training and 30% test

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def find_nn_size(n1, n2, act='relu', epoch=50):
    start = time.time()
    model = Sequential()
    model.add(Dense(n1, input_dim=30, activation=act))
    model.add(Dense(n2, activation=act))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])
    
    model.fit(X_train, y_train, epochs=epoch, batch_size=10,verbose=0)
    metrics = model.evaluate(X_test, y_test)
    
    end = time.time()
    delta_t = round(end-start, 1)
    print('Network: {n1}/{n2}, activation: {act},accuracy: {acc}, loss:{loss}, time:{time}'.format(n1=n1,n2=n2,loss=metrics[0], act=act,acc=metrics[1], time=delta_t))
    return metrics, delta_t
#
print('finding network size' )
df = pd.DataFrame()
for n1 in [5, 10 ,20]:
    for n2 in [ 5, 10, 20]:
        metrics, delta_t  = find_nn_size(n1, n2)
        df.loc[str(n1)+'/'+str(n2),'loss'] = metrics[0]
        df.loc[str(n1)+'/'+str(n2),'acc'] = metrics[1]
        df.loc[str(n1)+'/'+str(n2),'auc'] = metrics[2]
        df.loc[str(n1)+'/'+str(n2),'time'] =delta_t
df = df.round(3)        
print(df)

def nn(n1, n2, act='relu',validation_split=0.33, epoch=50):
    model = Sequential()
    model.add(Dense(n1, input_dim=30, activation=act))
    model.add(Dense(n2, activation=act))
    model.add(Dense(1, activation='sigmoid'))
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','roc_auc'])
    
    history = model.fit(X_undersample, y_undersample, epochs=epoch,validation_split=validation_split, batch_size=10,verbose=0)
    return model, history
#
print('Training model')
n1=20
n2=5
model, history = nn(n1,n2,act='relu',epoch=100 )

def plot_history(history):
    fig, ax = plt.subplots(2, 1)
    # summarize history for accuracy
    ax[0].plot(history.history['acc'])
    ax[0].plot(history.history['val_acc'])
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'test'], loc='upper left')
    fig.tight_layout()
    fig.savefig('nn_train.png')    
    
plot_history(history)   
 
final_model, history = nn(n1,n2,act='relu',epoch=20)

print('  Train accuracy: ', history.history['acc'][-1])
print('  Test accuracy: ', history.history['val_acc'][-1])
print(' train loss: ', history.history['loss'][-1])
print(' Test loss: ', history.history['val_loss'][-1])

