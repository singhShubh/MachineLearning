import numpy as np
import pandas as pd
from sklearn import neighbors,preprocessing,model_selection

df = pd.read_csv('breast-cancer-wisconsin.data')
#print(df.head())

df.drop(['id'],axis=1,inplace=True)
#axis=1 to apply a method across each row, or to the column labels.

df.replace('?',-99999, inplace=True)
X = df.drop(['class'],1)
Y = df['class']

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y, test_size=0.2)
model = neighbors.KNeighborsClassifier()
model.fit(X_train,Y_train)

accuracy = model.score(X_test,Y_test)
print('Testing accuracy: ',accuracy)

# Predictions on new data
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = model.predict(example_measures)
print('Predictions: ',prediction)