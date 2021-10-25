import pandas as pd
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

colnames = ["pregnant", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age", "label"]
df = pd.read_csv("data.csv", names = colnames).iloc[1:]
print(df.head())

features = ["pregnant", "insulin", "bmi", "age", "glucose", "bp", "pedigree"]
X = df[features]
y = df.label

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 1)
clf = DecisionTreeClassifier()
clf = clf.fit(xtrain, ytrain)

ypredict = clf.predict(xtest)
print("Accuracy is: ", metrics.accuracy_score(ytest, ypredict))

#visualizing the data
#where we store the data from our decision tree classifier as text
dotdata = StringIO()
export_graphviz(clf, out_file = dotdata, filled = True, rounded = True, special_characters = True, feature_names = features, class_name = ["0", "1"])
print(dotdata.getvalue())