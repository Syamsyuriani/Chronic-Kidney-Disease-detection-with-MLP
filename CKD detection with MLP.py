#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from matplotlib.pyplot import figure
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD

#Upload data
dt = pd.read_csv('https://raw.githubusercontent.com/Syamsyuriani/Handling-missing-values-in-the-CKD-dataset/main/Kidney_disease(Processed).csv')

#Feature Selection
Y = dt.classification
X = dt.drop(columns = ['classification'], axis = 1)

def processSubset(feature_set):
    model = sm.OLS(Y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - Y) ** 2).sum()
    MSE = RSS / (X.shape[0]-2)
    Cp = (1/X.shape[0]) * (RSS + 2 * len(feature_set) * MSE)
    return {"model":regr, "Cp":Cp}

def backward(predictors):
    result = []   
    for col in itertools.combinations(predictors, len(predictors)-1):
        result.append(processSubset(col))    
    models = pd.DataFrame(result)
    
    # Choose the model with the smallest Cp value
    best_model = models.loc[models['Cp'].argmin()]
    return best_model

model_bwd = pd.DataFrame(columns=["Cp", "model"], index = range(1,len(X.columns))) 
predictors = X.columns
while(len(predictors) > 1):  
    model_bwd.loc[len(predictors)-1] = backward(predictors)
    predictors = model_bwd.loc[len(predictors)-1]["model"].model.exog_names 
 
    print('\nSelected model with {} variable:'.format(len(predictors)))
    print(pd.DataFrame({'variable': predictors, 'Coefficient': list(model_bwd.loc[len(predictors), "model"].params)}))

#Compare the Cp value of the selected model
feat23 = ['age', 'bp', 'sg', 'al', 'su', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc',
       'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet',
       'pe', 'ane']
pd.set_option('max_colwidth', None)
out = pd.DataFrame(columns=["Cp", "model"], index = range(1,len(X.columns)))
predictors = X.columns
model23 = backward(predictors)
feat = []
while(len(predictors) > 1):
    out.loc[len(predictors)-1] = backward(predictors)
    predictors = out.loc[len(predictors)-1]["model"].model.exog_names
    feat.append(predictors)
feat.reverse()
feat.append(feat23)
cp = list(out['Cp'])
cp.append(model23['Cp'])
directory = pd.DataFrame({'Cp': cp,'Feature(s)': feat},index=range(1,24))
directory

#Select model with smallest Cp value
directory.loc[directory['Cp'].argmin() + 1]

#Replace independent variables with selected features
X = dt.drop(columns=['age', 'su', 'pc', 'pcc', 'ba', 'sod', 'pot', 'appet', 'pe'])

#Create new variables to indicate numeric and category features
cat_feat = ['al', 'htn', 'dm', 'cad', 'ane']
X_num = X.drop(columns=cat_feat)
X_cat = X[cat_feat]

#Numeric features standardization
scaler = StandardScaler()
scaler.fit(X_num)
X_num = scaler.transform(X_num)

X = pd.concat([X_num, X_cat], axis=1)
X.rename(columns=({ 0: 'bp', 1: 'sg', 2: 'bgr', 3: 'bu', 4: 'sc', 5: 'hemo', 6: 'pcv', 7: 'wc', 8: 'rc'}),inplace=True)

#one hot encoding on class features
Y = LabelEncoder().fit_transform(Y)

#Data split (data train and test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

#Model Building
random.seed(69)
def create_model(learn_rate, hidden_layers, neurons):
  model = Sequential()
  model.add(Dense(14))
  for i in range(hidden_layers):
    model.add(Dense(neurons, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  optimizer = SGD(learning_rate=learn_rate)
  model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
  return model
 
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

param_grid = {"hidden_layers": [1,2,3,4,5], "neurons": [1,2,3,5,6,7,8,9,10,11,12,13,14], "learn_rate": [0.0001, 0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
grid_result = grid.fit(X_train, y_train)
 
print("Best accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Accuracy: %f (%f) with: %r" % (mean, stdev, param))

#Model evaluation
y_pred = grid_result.predict(X_test)
print('Accuracy:',metrics.accuracy_score(y_pred,y_test))
print('Precision:',metrics.precision_score(y_pred,y_test))
print('Recall:',metrics.recall_score(y_pred,y_test))
