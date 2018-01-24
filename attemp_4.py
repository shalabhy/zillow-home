
import pandas as pd
import numpy as np

import gc

properties1 = pd.read_csv('properties_2016.csv')
train = pd.read_csv('train_2016_v2.csv')


column_names = ['parcelid', 'airconditioningtypeid',
       'architecturalstyletypeid', 'basementsqft', 'bathroomcnt', 'bedroomcnt',
       'buildingclasstypeid', 'buildingqualitytypeid', 'calculatedbathnbr',
       'decktypeid', 'finishedfloor1squarefeet',
       'calculatedfinishedsquarefeet', 'finishedsquarefeet12',
       'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50',
       'finishedsquarefeet6', 'fips', 'fireplacecnt', 'fullbathcnt',
       'garagecarcnt', 'garagetotalsqft', 'hashottuborspa',
       'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet',
       'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
       'propertycountylandusecode', 'propertylandusetypeid',
       'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity',
       'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt',
       'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid',
       'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt',
       'numberofstories', 'fireplaceflag', 'structuretaxvaluedollarcnt',
       'taxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt',
       'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear',
       'censustractandblock']

for column in column_names:
    print("% of null values",column," = ",(pd.isnull(properties1[column]).sum()/2985217)*100)


for column in column_names:
    if((pd.isnull(properties1[column]).sum()/2985217)*100 > 90):
        properties1 = properties1.drop([column], axis = 1)
        
    else:
        pass
properties.columns

column_names2 = ['parcelid', 'airconditioningtypeid', 'bathroomcnt', 'bedroomcnt',
       'buildingqualitytypeid', 'calculatedbathnbr',
       'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'fips',
       'fireplacecnt', 'fullbathcnt', 'garagecarcnt', 'garagetotalsqft',
       'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet',
       'poolcnt', 'pooltypeid7', 'propertycountylandusecode',
       'propertylandusetypeid', 'propertyzoningdesc', 'rawcensustractandblock',
       'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip',
       'roomcnt', 'threequarterbathnbr', 'unitcnt', 'yearbuilt',
       'numberofstories', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
       'assessmentyear', 'landtaxvaluedollarcnt', 'taxamount',
       'censustractandblock']

for column in column_names2:
    print("% of null values",column," = ",(pd.isnull(properties1[column]).sum()/2985217)*100)

df = train.merge(properties1, how = 'left',on = 'parcelid')

from scipy.stats import mode
df['buildingqualitytypeid'].fillna(mode(df['buildingqualitytypeid']).mode[0],inplace = True)
df['calculatedbathnbr'].fillna(mode(df['calculatedbathnbr']).mode[0],inplace = True)
df['calculatedfinishedsquarefeet'].fillna(mode(df['calculatedfinishedsquarefeet']).mode[0],inplace = True)
df['finishedsquarefeet12'].fillna(mode(df['finishedsquarefeet12']).mode[0],inplace = True)
df['fullbathcnt'].fillna(mode(df['fullbathcnt']).mode[0],inplace = True)
df['garagecarcnt'].fillna(mode(df['garagecarcnt']).mode[0],inplace = True)
df['garagetotalsqft'].fillna(mode(df['garagetotalsqft']).mode[0],inplace = True)
df['heatingorsystemtypeid'].fillna(mode(df['heatingorsystemtypeid']).mode[0],inplace = True)
df['lotsizesquarefeet'].fillna(mode(df['lotsizesquarefeet']).mode[0],inplace = True)
df['propertycountylandusecode'].fillna('0100', inplace = True)
df['propertyzoningdesc'].fillna('LAR3',inplace = True)
df['regionidcity'].fillna(mode(df['regionidcity']).mode[0],inplace = True)

df.drop(['regionidneighborhood'],axis = 1, inplace = True)

column_names2.remove('regionidneighborhood')

df['regionidzip'].fillna(mode(df['regionidzip']).mode[0],inplace = True)

df['unitcnt'].fillna(mode(df['unitcnt']).mode[0],inplace = True)
df['yearbuilt'].fillna(mode(df['yearbuilt']).mode[0],inplace = True)
df['unitcnt'].fillna(mode(df['unitcnt']).mode[0],inplace = True)

df.drop(['numberofstories'],axis = 1, inplace = True)

column_names2.remove('numberofstories')

from numpy import mean

df['structuretaxvaluedollarcnt'].fillna(mean(df['structuretaxvaluedollarcnt']),inplace = True)
#redo the above step
df['taxvaluedollarcnt'].fillna(mean(df['taxvaluedollarcnt']),inplace = True)
df['landtaxvaluedollarcnt'].fillna(mean(df['landtaxvaluedollarcnt']),inplace = True)
df['taxamount'].fillna(mean(df['taxamount']),inplace = True)
df['censustractandblock'].fillna(mean(df['censustractandblock']),inplace = True)
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ['Count','Column Type']


df_train= df.drop(['parcelid','transactiondate'],axis = 1)
#new

gc.collect()

#make test data 
del column_names
column_names2.remove('logerror')
column_names2.remove('transactiondate')
column_names2.remove('poolcnt')
column_names2.remove('pooltypeid7')
column_names2.remove('threequarterbathnbr')
properties1 = properties[column_names2]    
del properties
for column in column_names2:
    print("% of null values",column," = ",(pd.isnull(properties1[column]).sum()/2985217)*100)

from scipy.stats import mode

#properties1.drop(['threequarterbathnbr'],inplace = True)
properties1['regionidneighborhood'].fillna(properties1['regionidneighborhood'].value_counts().argmax(),inplace = True)
#properties1['threequarterbathnbr'].fillna(properties1['regionidneighborhood'].value_counts().argmax(),inplace = True)
properties1['numberofstories'].fillna(properties1['numberofstories'].value_counts().argmax(),inplace = True)

properties1['calculatedbathnbr'].fillna(properties1['calculatedbathnbr'].value_counts().argmax(),inplace = True)
properties1['fireplacecnt'].fillna(properties1['fireplacecnt'].value_counts().argmax(),inplace = True)
#properties1.drop(['poolcnt'],axis = 1,inplace = True)
#properties1.drop(['pooltypeid7'],axis = 1,inplace = True)
properties1['buildingqualitytypeid'].fillna(properties1['buildingqualitytypeid'].value_counts().argmax(),inplace = True)
properties1['calculatedbathnbr'].fillna(properties1['calculatedbathnbr'].value_counts().argmax(),inplace = True)
properties1['calculatedfinishedsquarefeet'].fillna(properties1['calculatedfinishedsquarefeet'].value_counts().argmax(),inplace = True)
properties1['finishedsquarefeet12'].fillna(properties1['finishedsquarefeet12'].value_counts().argmax(),inplace = True)
properties1['fullbathcnt'].fillna(properties1['fullbathcnt'].value_counts().argmax(),inplace = True)
properties1['garagecarcnt'].fillna(properties1['garagecarcnt'].value_counts().argmax(),inplace = True)
properties1['garagetotalsqft'].fillna(properties1['garagetotalsqft'].value_counts().argmax(),inplace = True)
properties1['heatingorsystemtypeid'].fillna(properties1['heatingorsystemtypeid'].value_counts().argmax(),inplace = True)
properties1['lotsizesquarefeet'].fillna(properties1['lotsizesquarefeet'].value_counts().argmax(),inplace = True)
properties1['propertycountylandusecode'].fillna('0100', inplace = True)
properties1['propertyzoningdesc'].fillna('LAR3',inplace = True)
properties1['regionidcity'].fillna(properties1['regionidcity'].value_counts().argmax(),inplace = True)

#properties1.drop(['regionidneighborhood'],axis = 1, inplace = True)

column_names2.remove('regionidneighborhood')

properties1['regionidzip'].fillna(properties1['regionidzip'].value_counts().argmax(),inplace = True)

sns.countplot(x = 'unitcnt', data = properties1)
properties1['unitcnt'].fillna(properties1['unitcnt'].value_counts().argmax(),inplace = True)
properties1['yearbuilt'].fillna(properties1['yearbuilt'].value_counts().argmax(),inplace = True)
#properties1['unitcnt'].fillna(properties1['unitcnt'].value_counts().argmax(),inplace = True)

proerties1.drop(['numberofstories'],axis = 1, inplace = True)

column_names2.remove('numberofstories')

from numpy import mean

properties1['structuretaxvaluedollarcnt'].fillna(mean(properties1['structuretaxvaluedollarcnt']),inplace = True)
#redo the above step
properties1['taxvaluedollarcnt'].fillna(mean(properties1['taxvaluedollarcnt']),inplace = True)
properties1['landtaxvaluedollarcnt'].fillna(mean(properties1['landtaxvaluedollarcnt']),inplace = True)
properties1['taxamount'].fillna(mean(properties1['taxamount']),inplace = True)
properties1['censustractandblock'].fillna(mean(properties1['censustractandblock']),inplace = True)

properties1['airconditioningtypeid'].fillna(properties1['airconditioningtypeid'].value_counts().argmax(),inplace = True)
properties1['bedroomcnt'].fillna(properties1['bedroomcnt'].value_counts().argmax(),inplace = True)
properties1['bathroomcnt'].fillna(properties1['bathroomcnt'].value_counts().argmax(),inplace = True)

properties1['fips'].fillna(properties1['fips'].value_counts().argmax(),inplace = True)
properties1['latitude'].fillna(properties1['latitude'].value_counts().argmax(),inplace = True)
properties1['longitude'].fillna(properties1['longitude'].value_counts().argmax(),inplace = True)

properties1['propertylandusetypeid'].fillna(properties1['propertylandusetypeid'].value_counts().argmax(),inplace = True)
properties1['rawcensustractandblock'].fillna(properties1['rawcensustractandblock'].value_counts().argmax(),inplace = True)
properties1['regionidcounty'].fillna(properties1['regionidcounty'].value_counts().argmax(),inplace = True)

properties1['roomcnt'].fillna(properties1['roomcnt'].value_counts().argmax(),inplace = True)
properties1['assessmentyear'].fillna(properties1['assessmentyear'].value_counts().argmax(),inplace = True)
properties1['buildingqualitytypeid'].fillna(properties1['buildingqualitytypeid'].value_counts().argmax(),inplace = True)
properties1['poolcnt'].fillna(properties1['poolcnt'].value_counts().argmax(),inplace = True)
properties1['pooltypeid7'].fillna(properties1['pooltypeid7'].value_counts().argmax(),inplace = True)
properties1['threequarterbathnbr'].fillna(properties1['threequarterbathnbr'].value_counts().argmax(),inplace = True)



gc.collect()

properties1.drop(['parcelid'],axis =1, inplace = True)
df.drop(['parcelid'],axis =1, inplace = True)
df_y = df['logerror']
df.drop(['logerror'],axis = 1, inplace = True)
#####

trainX = np.zeros(shape=(90275,37))
trainy = np.zeros(shape =(90275,1))
testX = np.zeros(shape= (2985217,37))

from sklearn.preprocessing import LabelEncoder
encode =  LabelEncoder()
df.iloc[:,21] = encode.transform(df.iloc[:,21])
df.iloc[:,19] = encode.transform(df.iloc[:,19])
properties1.iloc[:,21] = encode.fit_transform(properties1.iloc[:,21])
properties1.iloc[:,19] = encode.fit_transform(properties1.iloc[:,19])
properties1.drop(['parcelid'],axis = 1,inplace = True)



for i in range(37):
    trainX[:,i] = df.iloc[:,i]
#for test

for i in range(37):
    testX[:,i] = properties1.iloc[:,i]
    
    
trainy[:,0] = df_y

from sklearn.preprocessing import StandardScaler
scalerx = StandardScaler()

trainX = scalerx.fit_transform(trainX)
#
testX = scalerx.transform(testX)
#

from sklearn.metrics import mean_absolute_error

#trying keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'tanh', input_dim = 37))

# Adding the second hidden layer

classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 40, init = 'uniform', activation = 'tanh'))

classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'tanh'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'tanh'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'tanh'))

# Compiling the ANN
classifier.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(trainX, trainy, batch_size = 100, nb_epoch = 10)

train_pred = classifier.predict(trainX)
mean_absolute_error(trainy,train_pred)

test_pred1 = classifier.predict(testX)


sample = pd.read_csv('sample_submission.csv')

del properties1

for i in range(1,7):
    sample.iloc[:,i] = test_pred1

sample.to_csv('result.csv', index = False)
 
sample.to_csv('sol1.csv',index = False)   

