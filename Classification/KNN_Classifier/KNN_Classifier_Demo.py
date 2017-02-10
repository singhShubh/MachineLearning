import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import KNN_Classifier


style.use('fivethirtyeight')

#----------------------------------Creating train_data---------------------------------

dataset = {'r':[[1,2],[2,3],[3,4],[4,1]], 'g':[[9,10],[10,12],[11,15],[13,14]]}

#--------------------------------------Test data---------------------------------------
predict = [3,6]

#-----------------------------Visualising train and test data--------------------------
plt.figure(1)
plt.title('Train-Test Data')
[[plt.scatter(entry[0],entry[1],color=class_name, s=100) for entry in dataset[class_name]] for class_name in dataset]
plt.scatter(predict[0],predict[1],color='b', s=100)

#-----------------Getting the class of the test data using KNN_Classifier--------------
votes = KNN_Classifier.k_nearest_neighbours(dataset,predict,3)

#----------------------------Displaying the classifier output--------------------------
plt.figure(2)
plt.title('KNN_Classifier result')
[[plt.scatter(entry[0],entry[1],color=class_name, s=100) for entry in dataset[class_name]] for class_name in dataset]
plt.scatter(predict[0],predict[1],color=votes, s=100)


plt.show()
