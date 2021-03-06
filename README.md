# Chronic-Kidney-Disease-detection-with-MLP
Kidney is an important organ because it is the principal component of the human excretory system, mainly urine.
As a result, we must take care of our kidneys to prevent them from being infected with serious diseases such as chronic kidney disease (CKD).
Some of the causes of CKD include long-term use of various types of medicines, as well as a poor diet, such as eating too many salty foods or drinking too little mineral water. Other variables, such as cigarette smoke or breathing excessive air pollution, are also present.

CKD is a disease that affects a large number of people across the world. In 2017, there were 697.5 million people having CKD, and 1.2 million people died as a result of the disease. Therefore, CKD is the world's 12th largest cause of death. In 2010, more than 2 million people throughout the world received dialysis or kidney transplantation treatment, implying that roughly 10% of the population received treatment and the remainder perished each year due to a lack of services. It may be deduced from this that CKD, which was the 18th leading cause of mortality in the world in 2010, increased to 12th in 2017 [[1]](https://www.nature.com/articles/s41581-020-0268-7#:~:text=In%202017%2C%20CKD%20resulted%20in,%25%20of%20all%2Dcause%20mortality).

Because there are no symptoms in early-stage CKD, testing is the only way to determine if a patient has kidney disease. If a person has CKD and does not seek treatment or is unaware that their kidney function has decreased, complications or ESRD, or end-stage renal disease, are highly likely to occur in the future. If complications or ESRD occur, it is extremely serious and can lead to death. So that early detection of CKD is necessary to limit the number of patients, so that treatment and follow-up may begin right away, and complications like ESRD can be avoided [[2]](https://thesai.org/Publications/ViewPaper?Volume=10&Issue=8&Code=IJACSA&SerialNo=13).

We can utilize machine learning to assist clinicians in detecting CKD fast for big volumes of data. We employ a multilayer perceptron (MLP) in our work to diagnose CKD early. This work's details can be found at this [URL](https://github.com/Syamsyuriani/Chronic-Kidney-Disease-detection-with-MLP/blob/main/Mathematical-Modelling-of-Chronic-Kidney-Disease-with-MLP.pdf), and the source code at this [URL](https://github.com/Syamsyuriani/Chronic-Kidney-Disease-detection-with-MLP/blob/main/CKD%20detection%20with%20MLP.py)

## About the Data
We use data from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease). This data comes from the Apollo Hospital in India. There are 25 features in this data, including the categorization class ckd or not ckd. This data is from 400 patients with early stage CKD, 250 of those has CKD and 150 not CKD. The data has a missing value so that the missing value is handled first. The process handling missing values in this data can be seen in this [source](https://github.com/Syamsyuriani/Handling-missing-values-in-the-CKD-dataset).
```python
dt = pd.read_csv('https://raw.githubusercontent.com/Syamsyuriani/Handling-missing-values-in-the-CKD-dataset/main/Kidney_disease(Processed).csv')
```
![image](https://user-images.githubusercontent.com/72261134/151430339-feea6c86-48c0-4893-b5b9-3fcbae5835ae.png)

## Feature Selection
After handling the missing value, we perform feature selection using the backward selection technique by paying attention to the [Cp value](https://booksvooks.com/nonscrolablepdf/an-introduction-to-statistical-learning-with-applications-in-r-pdf.html). The selected feature is a combination of features that produces the smallest Cp value.
```python
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

X = dt.drop(columns=['age', 'su', 'pc', 'pcc', 'ba', 'sod', 'pot', 'appet', 'pe'])
```
![image](https://user-images.githubusercontent.com/72261134/151427230-53231bc8-14c1-479f-9173-81af1c670065.png)

The smallest Cp value is achieved by a combination of 14 features, including bp, sg, al, bgr, bu, sc, hemo, pcv, wc, rc, htn, dm, cad, and ane, as shown in the figure above.

## Data Standardization
After performing feature selection, the next step is to standardize the data so that the model can learn faster and improve model performance. Data standardization is done by making the numerical features normally distributed with mean 0 and standard deviation 1.
```python
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
```
The next step is to split the data into train data and test data. Train data is used to train the model which is then evaluated using test data. We divide the data into 80% training data and 20% test data.
```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
```
## Model Building
To build the model, we use cross validation with 10-folds to estimate the model performance.
We also perform hyperparameter tuning with grid search technique to find the best combination of hyperparameters for optimal performance.
Candidate hyperparameter values that must be optimized are shown in the following table.
We train the model using epoch 10, batch size 32, and SGD learning method.

![image](https://user-images.githubusercontent.com/72261134/151433149-d17ee482-2f36-4c4c-88d6-688a2cdfe878.png)

```python
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
```

## Model Evaluation
The last step is to evaluate the model that has been buit. We We focus on three evaluation criteria: accuracy, precision, and recall.

```python
y_pred = grid_result.predict(X_test)
print('Accuracy:',metrics.accuracy_score(y_pred,y_test))
print('Precision:',metrics.precision_score(y_pred,y_test))
print('Recall:',metrics.recall_score(y_pred,y_test))
```
After 10 repetitions, the average performance of the model as follows.
```
Accuracy: 0.9845070422535211
Precision: 0.9881294944391446
Recall: 0.9852268142058274
```
