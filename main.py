import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

columns = ["cap-shape","cap-surface","cap-color","bruises%3F"
     ,"odor","gill-attachment","gill-spacing","gill-size","gill-color",
     "stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-above-ring",
     "stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type",
     "veil-color","ring-number","ring-type","spore-print-color","population","habitat","class"]

def getData():
    data = pd.read_csv("mushroom.csv")
    labelEncoder = LabelEncoder()

    for i in columns:
        data[i] = labelEncoder.fit_transform(data[i])

    X = data.drop(['class'], axis=1)
    Y = data['class']

    print("Data has been loaded...\n")
    return train_test_split(X, Y, test_size=0.2, random_state=101)

def CreateChart(loss_history, label_name, fin, name="dafault.png"):
    #plt.plot(range(len(loss_history)), loss_history)
    sub = plt.subplot()
    sub.plot(range(len(loss_history)), loss_history, label=label_name)
    sub.legend()
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    if(fin == True):
        plt.savefig(name)

def printTestValues(X_train, X_test, Y_train, Y_test):
    classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    classifier.fit(X_train, Y_train)
    prediction = classifier.predict(X_test)
    print("Train set values:\n")
    print(classification_report(Y_test, prediction))


def ADAPredict(X_train, X_test, Y_train, Y_test):

    boostClasifier = AdaBoostClassifier(n_estimators = 10)
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
    print("Test set values predicted with MLP basic setup(2 neurons, 10 iterations):\n")
    print(classification_report(Y_pred, Y_test))
    CreateChart(loss_history, "Basic", False)

def MLPPredictPrecise(X_train, X_test, Y_train, Y_test):
    MLPclassifier = MLPClassifier(hidden_layer_sizes = (20,), max_iter = 100, verbose = 1)
    history = MLPclassifier.fit(X_train, Y_train)
    loss_history = history.loss_curve_
    Y_pred = MLPclassifier.predict(X_test)
    print("Test set values predicted with MLP precise setup(15 neurons, 50 iterations):\n")
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


def makePrediction():
    X_train, X_test, Y_train, Y_test = getData()

    printTestValues(X_train, X_test, Y_train, Y_test)

    ADAPredict(X_train, X_test, Y_train, Y_test)

    MLPPredictBasic(X_train, X_test, Y_train, Y_test)
    MLPPredictPrecise(X_train, X_test, Y_train, Y_test)
    MLPPredictSGD(X_train, X_test, Y_train, Y_test)
    MLPPredictSGDOvertrained(X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":
    makePrediction()