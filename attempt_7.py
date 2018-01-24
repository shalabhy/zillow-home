# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 08:28:41 2017

@author: CG-DTE
"""
import pandas as pd 
import numpy as np 
import matplotlib as plt
import seaborn as sns

properties1 =pd.read_csv('properties_2016.csv')
i = 0
cols1 = []
for col in cols:
    print(col,"        ",(properties1[col].isnull().sum()/2985217)*100)
    if properties1[col].isnull().sum() > 0:
        cols1.append(col)
    else:
        pass
null_val = np.zeros((49,2))
null_val = pd.DataFrame(null_val)
null_val.columns = ['columns','% missing']

null_val.to_csv('missing values percentage.csv', index = False)

# first lets fill na values in columns where na can be replaced by median or mean
#median
properties1['bathroomcnt'].fillna(properties1['bathroomcnt'].value_counts().argmax(),inplace = True)
properties1['bedroomcnt'].fillna(properties1['bedroomcnt'].value_counts().argmax(),inplace = True)
properties1['fireplacecnt'].fillna(properties1['fireplacecnt'].value_counts().argmax(),inplace = True)
properties1['hashottuborspa'].fillna(properties1['hashottuborspa'].value_counts().argmax(),inplace = True)
properties1['heatingorsystemtypeid'].fillna(properties1['heatingorsystemtypeid'].value_counts().argmax(),inplace = True)
properties1['pooltypeid7'].fillna(properties1['pooltypeid7'].value_counts().argmax(),inplace = True)
properties1['propertylandusetypeid'].fillna(properties1['propertylandusetypeid'].value_counts().argmax(),inplace = True)
properties1['propertycountylandusecode'].fillna(properties1['propertycountylandusecode'].value_counts().argmax(),inplace = True)
properties1['propertyzoningdesc'].fillna(properties1['propertyzoningdesc'].value_counts().argmax(),inplace = True)
properties1['rawcensustractandblock'].fillna(properties1['rawcensustractandblock'].value_counts().argmax(),inplace = True)
properties1['regionidcity'].fillna(properties1['regionidcity'].value_counts().argmax(),inplace = True)
properties1['regionidcounty'].fillna(properties1['regionidcounty'].value_counts().argmax(),inplace = True)
properties1['regionidzip'].fillna(properties1['regionidzip'].value_counts().argmax(),inplace = True)
properties1['roomcnt'].fillna(properties1['roomcnt'].value_counts().argmax(),inplace = True)
properties1['yearbuilt'].fillna(properties1['yearbuilt'].value_counts().argmax(),inplace = True)
properties1['taxdelinquencyyear'].fillna(properties1['taxdelinquencyyear'].value_counts().argmax(),inplace = True)
properties1['censustractandblock'].fillna(properties1['censustractandblock'].value_counts().argmax(),inplace = True)
properties1['fips'].fillna(properties1['fips'].value_counts().argmax(),inplace = True)
properties1['assessmentyear'].fillna(properties1['assessmentyear'].value_counts().argmax(),inplace = True)
properties1['fullbathcnt'].fillna(properties1['fullbathcnt'].value_counts().argmax(),inplace = True)

#mean
from numpy import mean
properties1['calculatedfinishedsquarefeet'].fillna(mean(properties1['calculatedfinishedsquarefeet']), inplace = True)
properties1['finishedsquarefeet12'].fillna(mean(properties1['finishedsquarefeet12']), inplace = True)
properties1['latitude'].fillna(mean(properties1['latitude']), inplace = True)
properties1['longitude'].fillna(mean(properties1['longitude']), inplace = True)
properties1['lotsizesquarefeet'].fillna(mean(properties1['lotsizesquarefeet']), inplace = True)
properties1['structuretaxvaluedollarcnt'].fillna(mean(properties1['structuretaxvaluedollarcnt']), inplace = True)
properties1['taxvaluedollarcnt'].fillna(mean(properties1['taxvaluedollarcnt']), inplace = True)
properties1['taxamount'].fillna(mean(properties1['taxamount']), inplace = True)
properties1['landtaxvaluedollarcnt'].fillna(mean(properties1['landtaxvaluedollarcnt']), inplace = True)

#removing cols with more than 99% missing 
properties1.drop(['taxdelinquencyflag'], axis =1, inplace = True)
properties1.drop(['fireplaceflag'], axis =1, inplace = True)
properties1.drop(['yardbuildingsqft26'], axis =1, inplace = True)
properties1.drop(['typeconstructiontypeid'], axis =1, inplace = True)
properties1.drop(['storytypeid'], axis =1, inplace = True)
properties1.drop(['poolsizesum'], axis =1, inplace = True)
properties1.drop(['finishedsquarefeet6'], axis =1, inplace = True)
properties1.drop(['finishedsquarefeet13'], axis =1, inplace = True)
properties1.drop(['decktypeid'], axis =1, inplace = True)
properties1.drop(['buildingclasstypeid'], axis =1, inplace = True)
properties1.drop(['basementsqft'], axis =1, inplace = True)
properties1.drop(['architecturalstyletypeid'], axis =1, inplace = True)


properties1.to_csv('new_property.csv', index = False)
noemptyvals = []
for col in cols:
    if properties1[col].isnull().sum() > 0 :
        pass
    else:
        noemptyvals.append(col)
        
#filling empty values using ANN     
trainx = properties1[noemptyvals]
trainy = properties1['airconditioningtypeid']
del(properties1)
trainx['airconditioningtypeid'] = trainy 

trainx = pd.read_csv('correctedfullvalueswithair.csv')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder1 = LabelEncoder()
onehot = OneHotEncoder(categorical_features = [0])
trainx['hashottuborspa'] = encoder.fit_transform(trainx['hashottuborspa'])
trainx.drop(['propertycountylandusecode'], axis =1, inplace = True)
trainx['propertyzoningdesc'] = encoder1.fit_transform(trainx['propertyzoningdesc'])
#trainx = onehot.fit_transform(trainx).toarray()


index = trainx[trainx['airconditioningtypeid'].isnull()].index
indexc = trainx[trainx['airconditioningtypeid'].isnull()== False].index
test = trainx.iloc[index,:]
train = trainx.iloc[indexc,:]

trainx.to_csv('correctedfullvalueswithair.csv', index = False)
import gc
del trainx
#trial
X = train.iloc[:,1:29]
Y = np.zeros((811519,1))
Y[:,0]= train.iloc[:,29]


Y = onehot.fit_transform(Y).toarray()

from sklearn.preprocessing import StandardScaler
scalex = StandardScaler()
X = scalex.fit_transform(X)
from sklearn.cross_validation import train_test_split
x_train,x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.4, random_state = 42)

# classifier

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes= (100,),activation="logistic", solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,max_iter= 200,random_state = 1)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
#test
X = test.iloc[:,1:29]
X = scalex.fit_transform(X)
prediction = classifier.predict(X)

#prediction = onehot.inverse_transform(prediction)
decoded = prediction.dot(onehot.active_features_).astype(int)

test.iloc[:,29] = decoded
test = test[['parcelid','airconditioningtypeid']]
train = train[['parcelid','airconditioningtypeid']]
result = pd.concat([test,train])
properties1 = pd.read_csv('new_property.csv')
properties1.drop(['airconditioningtypeid'], axis = 1, inplace = True)
properties1 = properties1.merge(result, how = 'left',on = 'parcelid')

properties1.to_csv('new_property.csv', index = False)
###################################################
properties1 = pd.read_csv('new_property.csv')
########
trainx = pd.read_csv('correctedfullvalueswithair.csv')
trainx.drop(['airconditioningtypeid'],axis = 1, inplace = True)

trainx['buildingqualitytypeid'] = properties1['buildingqualitytypeid']
trainx.drop(['propertycountylandusecode'], axis =1, inplace = True)

index = trainx[trainx['buildingqualitytypeid'].isnull()].index
indexc = trainx[trainx['buildingqualitytypeid'].isnull()== False].index
test = trainx.iloc[index,:]
train = trainx.iloc[indexc,:]
X = train.iloc[:,1:29]
Y = np.zeros((1938488,1))
Y[:,0]= train.iloc[:,29]
Y = onehot.fit_transform(Y).toarray()

from sklearn.preprocessing import StandardScaler
scalex = StandardScaler()
X = scalex.fit_transform(X)

#train 
''' 
classifier = MLPClassifier(hidden_layer_sizes= (100,),activation="relu", solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,max_iter= 200,random_state = 1)
classifier.fit(X,Y)'''

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'linear', input_dim = 28))

# Adding the second hidden layer

classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'linear'))
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'linear'))

# Adding the output layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'sigmoid'))
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9,nesterov = True)
# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, Y, batch_size = 1000, nb_epoch = 10)

#regressor.fit(trainX,trainy)
X1 = test.iloc[:,1:29]
X1 = scalex.fit_transform(X1)
train_pred = classifier.predict(X1)
train_pred[train_pred >= 0.5] = 1
train_pred[train_pred < 0.5] = 0

decoded = train_pred.dot(onehot.active_features_).astype(int)
decoded1 = Y.dot(onehot.active_features_).astype(int)

test.iloc[:,29] = decoded
test = test[['parcelid','buildingqualitytypeid']]
train = train[['parcelid','buildingqualitytypeid']]
result = pd.concat([test,train])
properties1 = pd.read_csv('new_property.csv')
properties1.drop(['buildingqualitytypeid'], axis = 1, inplace = True)
properties1 = properties1.merge(result, how = 'left',on = 'parcelid')
properties1.to_csv('new_property.csv', index = False)

####################################################
properties1 = pd.read_csv('new_property.csv')
trainx = pd.read_csv('correctedfullvalueswithair.csv')

trainx.drop(['propertycountylandusecode'], axis =1, inplace = True)
trainx['numberofstories'] = properties1['numberofstories']

index = trainx[trainx['numberofstories'].isnull()].index
indexc = trainx[trainx['numberofstories'].isnull()== False].index
test = trainx.iloc[index,:]
train = trainx.iloc[indexc,:]
X = test.iloc[:,1:31]
Y = np.zeros((682069,1))
Y[:,0]= train.iloc[:,31]
Y = onehot.fit_transform(Y).toarray()

X = scalex.transform(X)
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'linear', input_dim = 30))

# Adding the second hidden layer

classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'linear'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'linear'))

# Adding the output layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'sigmoid'))
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9,nesterov = True)
# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, Y, batch_size = 1000, nb_epoch = 10)
train_pred = classifier.predict(X)
train_pred[train_pred >= 0.5] = 1
train_pred[train_pred < 0.5] = 0
decoded = train_pred.dot(onehot.active_features_).astype(int)


test.iloc[:,31] = decoded
test = test[['parcelid','numberofstories']]
train = train[['parcelid','numberofstories']]
result = pd.concat([test,train])
properties1.drop(['numberofstories'], axis =1, inplace = True)
properties1 = properties1.merge(result, how = 'left',on = 'parcelid')

properties1.to_csv('new_property.csv',index = False)
######################################################
trainx = pd.read_csv('correctedfullvalueswithair.csv')

trainx.drop(['propertycountylandusecode'], axis =1, inplace = True)
trainx['buildingqualitytypeid'] = properties1['buildingqualitytypeid']
trainx['airconditioningtypeid'] = properties1['airconditioningtypeid']
trainx['threequarterbathnbr'] = properties1['threequarterbathnbr']

index = trainx[trainx['threequarterbathnbr'].isnull()].index
indexc = trainx[trainx['threequarterbathnbr'].isnull()== False].index
test = trainx.iloc[index,:]
train = trainx.iloc[indexc,:]
X = test.iloc[:,1:31]
Y = np.zeros((311631,1))
Y[:,0]= train.iloc[:,31]
Y = onehot.fit_transform(Y).toarray()
X = scalex.transform(X)

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'linear', input_dim = 30))

# Adding the second hidden layer

classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'linear'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'linear'))

# Adding the output layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'sigmoid'))
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9,nesterov = True)
# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, Y, batch_size = 1000, nb_epoch = 10)
train_pred = classifier.predict(X)
train_pred[train_pred >= 0.5] = 1
train_pred[train_pred < 0.5] = 0
decoded = train_pred.dot(onehot.active_features_).astype(int)

test.iloc[:,31] = decoded
test = test[['parcelid','threequarterbathnbr']]
train = train[['parcelid','threequarterbathnbr']]
result = pd.concat([test,train])
properties1.drop(['threequarterbathnbr'], axis =1, inplace = True)
properties1 = properties1.merge(result, how = 'left',on = 'parcelid')

properties1.to_csv('new_property.csv',index = False)

################################################################
trainx = pd.read_csv('correctedfullvalueswithair.csv')

trainx.drop(['propertycountylandusecode'], axis =1, inplace = True)
trainx['buildingqualitytypeid'] = properties1['buildingqualitytypeid']
trainx['airconditioningtypeid'] = properties1['airconditioningtypeid']
trainx['threequarterbathnbr'] = properties1['threequarterbathnbr']
trainx['garagecarcnt'] = properties1['garagecarcnt']
index = trainx[trainx['garagecarcnt'].isnull()].index
indexc = trainx[trainx['garagecarcnt'].isnull()== False].index
test = trainx.iloc[index,:]
train = trainx.iloc[indexc,:]
X = test.iloc[:,1:32]
Y = np.zeros((883267,1))
Y[:,0]= train.iloc[:,31]
Y = onehot.fit_transform(Y).toarray()
X = scalex.transform(X)

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'linear', input_dim = 31))

# Adding the second hidden layer

classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'linear'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'linear'))

# Adding the output layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'sigmoid'))
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9,nesterov = True)
# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, Y, batch_size = 1000, nb_epoch = 10)
train_pred = classifier.predict(X)
train_pred[train_pred >= 0.5] = 1
train_pred[train_pred < 0.5] = 0
decoded = train_pred.dot(onehot.active_features_).astype(int)

test.iloc[:,32] = decoded
test = test[['parcelid','garagecarcnt']]
train = train[['parcelid','garagecarcnt']]
result = pd.concat([test,train])
properties1.drop(['garagecarcnt'], axis =1, inplace = True)
properties1 = properties1.merge(result, how = 'left',on = 'parcelid')

properties1.to_csv('new_property.csv',index = False)


#################################################
# try 
properties1 = pd.read_csv('new_property.csv')
properties1.drop(cols1, axis =1, inplace = True)
train = pd.read_csv('train_2016_v2.csv')

df = train.merge(properties1, how = 'left',on = 'parcelid')

df.drop(['parcelid'],axis = 1, inplace =True)
df.drop(['transactiondate'],axis = 1, inplace = True)
properties1.drop(['parcelid'],axis =1 , inplace = True)
Y = np.zeros((90275,1))
Y[:,0] = df['logerror']
df.drop(['logerror'],axis = 1, inplace = True)
df.drop(['propertycountylandusecode'],axis = 1, inplace = True)
properties1.drop(['propertycountylandusecode'],axis = 1, inplace = True)
encode1 = LabelEncoder()
properties1.iloc[:,16] = encode1.fit_transform(properties1.iloc[:,16])


X = np.zeros((90275,34))
for i in range(34):
    X[:,i] = df.iloc[:,i] 
X = scalex.fit_transform(X)

from sklearn.metrics import mean_absolute_error

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'linear', input_dim = 34))

# Adding the second hidden layer


classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))

# Compiling the ANN
classifier.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, Y, batch_size = 100, nb_epoch = 10)

#regressor.fit(trainX,trainy)
train_pred = classifier.predict(X)
mean_absolute_error(Y,train_pred)