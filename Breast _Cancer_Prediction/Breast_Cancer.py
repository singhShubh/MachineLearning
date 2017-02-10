import pandas as pd
import KNN_Classifier

# Loading dataset
df = pd.read_csv('breast-cancer-wisconsin.data')
df.drop(['id'],axis=1,inplace=True)
df.replace('?',value=9999,inplace=True)

full_data = df.astype(float).values.tolist()

#Train_Test_split
test_size = 0.3
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}

for entry in train_data:
    train_set[entry[-1]].append(entry[:-1])

for entry in test_data:
    test_set[entry[-1]].append(entry[:-1])

# Checking the accuracy of our model

total = 0
correct = 0
total_cofidence=0
for class_name in test_set:
    for entry in test_set[class_name]:
        vote,confidence=KNN_Classifier.k_nearest_neighbours(train_set,entry,k=3)
        if vote == class_name:
            correct+=1
        total+=1
        total_cofidence+=confidence
print('Accuracy: ',correct/total*100)
print('Avg Confidence',total_cofidence/total*100)