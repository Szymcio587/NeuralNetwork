import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings("ignore")

columns = ["cap-shape","cap-surface","cap-color","bruises%3F"
     ,"odor","gill-attachment","gill-spacing","gill-size","gill-color",
     "stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-above-ring",
     "stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type",
     "veil-color","ring-number","ring-type","spore-print-color","population","habitat","class"]

data = pd.read_csv("mushroom.csv")

labelEncoder = LabelEncoder()

for i in columns:
    data[i] = labelEncoder.fit_transform(data[i])

correlation = data.corr()

plt.figure(figsize = (20,10),dpi = 200)
sns.heatmap(correlation,annot = True,cmap = 'viridis')

plt.figure(figsize=(20,10),dpi=200)
sns.barplot(data = correlation)
plt.xticks(rotation=90)

X = data.drop(['class'],axis=1)
Y = data['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 101)

classifier = RandomForestClassifier()

parameters = {'n_estimators':[100,200,300],
           'criterion' :['gini', 'entropy'],
           'max_features':['auto', 'sqrt', 'log2']}

grid=GridSearchCV(classifier,parameters)

grid.fit(X_train,Y_train)

classifier.fit(X_train,Y_train)

prediction = classifier.predict(X_test)

print(classification_report(Y_test,prediction))

plt.figure(figsize = (20,10),dpi = 200)
confusionMatrix = confusion_matrix(Y_test,prediction)
sns.heatmap(confusionMatrix,annot = True,cmap = 'mako')

boostClasifier = AdaBoostClassifier(n_estimators = 1)

boostClasifier .fit(X_train,Y_train)

BCPrediction = boostClasifier .predict(X_test)

boostClasifier .feature_importances_.argmax()

sns.countplot(data = data,x = 'gill-color',hue = 'class')

print(classification_report(BCPrediction,Y_test))