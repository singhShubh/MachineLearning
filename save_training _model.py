import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import quandl
import pickle


#-----------------------------------------------Getting data from Quandl-----------------------------------------
quandl.ApiConfig.api_key ="bLLLgt9DdMkHmWJawXHF"
df  = quandl.get('WIKI/GOOGL')
#----------------------------------------------------------------------------------------------------------------

#--------------------------------------------------Creating features---------------------------------------------
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
#print(df.head())
#----------------------------------------------------------------------------------------------------------------


#---------------------------------------------------Creating label-----------------------------------------------
forecast_col = 'Adj. Close'
forecast_out = 20      # we will try to predict the 20 days ahead stock prices(closing prices)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
#print(df.head())

#--------------------------------------Creating training and test set data---------------------------------------
X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X) # Feature scaling
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)

#------------------------------------------------------Training--------------------------------------------------
clf = LinearRegression()
clf.fit(X_train,y_train)
#----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------Saving the classifier--------------------------------------------

with open("linear_regession.pickle",'wb') as f:
    pickle.dump(obj=clf,file=f)
#----------------------------------------------------------------------------------------------------------------

#-----------------------------------------------Loading the classifier-------------------------------------------
# Just for making predictions, we can start from here by loading the classifier and making predictions.
# No need to train the classifier everytime we have to make a prediction.
pickle_in = open("linear_regession.pickle",'rb')
clf1 = pickle.load(pickle_in)
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------Checking the accuracy of our model--------------------------------------
accuracy = clf1.score(X_test,y_test)
print(accuracy)









