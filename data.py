import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import numpy as np
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.warn("deprecated", DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

np.random.seed(0)

FILE= 'creditcard.csv'
df = pd.read_csv(FILE)
label_column = 'Class'
y = df[label_column]
X = df.drop(label_column, axis=1)
    
class_weight = class_weight.compute_class_weight('balanced',
                                                np.unique(y),
                                                y)
def undersample(ret = ''):
    df = pd.read_csv(FILE)
    
    df = normallize(df)
    
    number_records_fraud = len(df[df.Class == 1])
    fraud_indices = np.array(df[df.Class == 1].index)
    
    # Picking the indices of the normal classes
    normal_indices = df[df.Class == 0].index
    
    # Out of the indices we picked, randomly select "x" number (number_records_fraud)
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud*3, replace = False)
    random_normal_indices = np.array(random_normal_indices)
    
    # Appending the 2 indices
    under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
    
    # Under sample dfset
    under_sample_df = df.iloc[under_sample_indices,:]
    if ret =='df':
        return under_sample_df
    
    X_undersample = under_sample_df.ix[:, under_sample_df.columns != 'Class']
    y_undersample = under_sample_df.ix[:, under_sample_df.columns == 'Class']
#    X_train, X_test, y_train, y_test = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=1) # 70% training and 30% test
    
    return X_undersample,y_undersample

def normallize(df):

    df['norm_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['norm_Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    df = df.drop(['Amount'],axis=1)
    df = df.drop(['Time'],axis=1)
    
    return df
    
def plot_fraud_hist(df):
    df = undersample(ret='df')
    fig, ax = plt.subplots(3,1, figsize=(5,8))
    count_classes = pd.value_counts(df['Class'], sort = True).sort_index()
    ax[0].bar(count_classes.index, count_classes)
    ax[0].set_xticks(range(2))
    ax[0].set_title("Fraud histogram")
    ax[0].set_xlabel("Class")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(df['norm_Time'])
    ax[1].set_title("Time histogram")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Frequency")
    
    ax[2].hist(df['norm_Amount'])
    ax[2].set_title("Amount histogram")
    ax[2].set_xlabel("Amount")
    ax[2].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig('df.png')


#plot_fraud_hist(df)


