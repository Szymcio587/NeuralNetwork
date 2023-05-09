import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns

warnings.filterwarnings("ignore")

def CreateHistograms(data):
    for col in data:
        if col == "class":
            continue
        data.pivot(columns='class', values=col).plot.hist(alpha=0.5, figsize=(5, 4))
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Occurances')
        plt.savefig(col)

    sns.kdeplot(data['stalk-root'], shade=True)
    plt.title('stalk-root')
    plt.xlabel('stalk-root')
    plt.ylabel('Density')
    plt.savefig('stalk-root')

def CheckData(data):
    print(data.describe())

    print(data['class'].value_counts())

    hist = data.hist
    print(hist)

    columns = ["cap-shape", "cap-surface", "cap-color", "bruises%3F"
        , "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color",
               "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-above-ring",
               "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
               "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat", "class"]

    labelEncoder = LabelEncoder()
    for i in columns:
        data[i] = labelEncoder.fit_transform(data[i])

    #CreateHistograms(data)


def GetData():
    data = pd.read_csv("mushroom.csv")

    CheckData(data)

    X = data.drop(['class'], axis=1)
    Y = data['class']

    print("Data has been loaded...\n")
    return train_test_split(X, Y, test_size=0.2, random_state=101)

def CreateChart(loss_history, label_name, fin, name="dafault.png"):
    sub = plt.subplot()
    sub.plot(range(len(loss_history)), loss_history, label=label_name)
    sub.legend()
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    if(fin == True):
        plt.savefig(name)

def PrintTestValues(X_train, X_test, Y_train, Y_test):
    classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    classifier.fit(X_train, Y_train)
    prediction = classifier.predict(X_test)
    print("Train set values:\n")
    print(classification_report(Y_test, prediction))


def ADAPredict(X_train, X_test, Y_train, Y_test):

    boostClasifier = AdaBoostClassifier(n_estimators = 2)
    boostClasifier.fit(X_train, Y_train)
    BCPrediction = boostClasifier.predict(X_test)
    boostClasifier.feature_importances_.argmax()
    print("Test set values predicted with ADA:\n")
    print(classification_report(BCPrediction, Y_test))

def MLPPredictBasic(X_train, X_test, Y_train, Y_test):
    MLPclassifier = MLPClassifier(hidden_layer_sizes = (2,), max_iter = 100, verbose = 1)
    history = MLPclassifier.fit(X_train, Y_train)
    loss_history = history.loss_curve_
    Y_pred = MLPclassifier.predict(X_test)
    print("Test set values predicted with MLP basic setup(2 neurons):\n")
    print(classification_report(Y_pred, Y_test))
    CreateChart(loss_history, "Basic", False)

def MLPPredictAverage(X_train, X_test, Y_train, Y_test):
    MLPclassifier = MLPClassifier(hidden_layer_sizes = (7,), max_iter = 100, verbose = 1)
    history = MLPclassifier.fit(X_train, Y_train)
    loss_history = history.loss_curve_
    Y_pred = MLPclassifier.predict(X_test)
    print("Test set values predicted with MLP average setup(7 neurons):\n")
    print(classification_report(Y_pred, Y_test))
    CreateChart(loss_history, "Average", False)

def MLPPredictPrecise(X_train, X_test, Y_train, Y_test):
    MLPclassifier = MLPClassifier(hidden_layer_sizes = (20,), max_iter = 100, verbose = 1)
    history = MLPclassifier.fit(X_train, Y_train)
    loss_history = history.loss_curve_
    Y_pred = MLPclassifier.predict(X_test)
    print("Test set values predicted with MLP precise setup(25 neurons):\n")
    print(classification_report(Y_pred, Y_test))
    CreateChart(loss_history, "Precise", False)

def MLPPredictSGD(X_train, X_test, Y_train, Y_test):
    MLPclassifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, verbose=2137, learning_rate_init=0.05,
                                  learning_rate='adaptive', solver='sgd')
    history = MLPclassifier.fit(X_train, Y_train)
    loss_history = history.loss_curve_
    Y_pred = MLPclassifier.predict(X_test)
    print("Test set values predicted with MLP custom setup(sgd solver, slow learning rate with adaptive abilities):\n")
    print(classification_report(Y_pred, Y_test))
    CreateChart(loss_history, "SDG", False)

def MLPPredictSGDOvertrained(X_train, X_test, Y_train, Y_test):
    MLPclassifier = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, verbose=2137, learning_rate_init=0.2, solver='sgd')
    history = MLPclassifier.fit(X_train, Y_train)
    loss_history = history.loss_curve_
    Y_pred = MLPclassifier.predict(X_test)
    print("Test set values predicted with MLP custom setup(sgd solver, too high learning rate for given problem):\n")
    print(classification_report(Y_pred, Y_test))
    CreateChart(loss_history, "Overtrained", True, "MLP_plots.png")


def MakePrediction():
    X_train, X_test, Y_train, Y_test = GetData()

    PrintTestValues(X_train, X_test, Y_train, Y_test)

    ADAPredict(X_train, X_test, Y_train, Y_test)

    MLPPredictBasic(X_train, X_test, Y_train, Y_test)
    MLPPredictAverage(X_train, X_test, Y_train, Y_test)
    MLPPredictPrecise(X_train, X_test, Y_train, Y_test)
    MLPPredictSGD(X_train, X_test, Y_train, Y_test)
    MLPPredictSGDOvertrained(X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":
    MakePrediction()