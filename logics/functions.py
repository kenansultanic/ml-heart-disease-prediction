import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics

from scipy import stats
from sklearn.model_selection import train_test_split as data_split


def train_test_split(data, test_size=0.2, random_state=42):
    np.random.seed(0)
    df_train, df_test = data_split(data, test_size=test_size, random_state=random_state)
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def print_metrics(y_test, y_test_pred, averaging="binary"):
    print("Accuracy:", metrics.accuracy_score(y_test, y_test_pred))
    print("Precision:", metrics.precision_score(y_test, y_test_pred, average=averaging))
    print("Recall:", metrics.recall_score(y_test, y_test_pred, average=averaging))
    print("F1 Score:", metrics.f1_score(y_test, y_test_pred, average=averaging))


def plot_confusion_matrix(y_test, y_test_pred):
    cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    class_names = ["No disease", "Has disease"]
    fig, ax = plt.subplots()

    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    ax.invert_yaxis()
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.title('ROC Curve')
    plt.show()
    

def plot_probability_distribution_hist(y_pred_proba):
    plt.hist(y_pred_proba, bins=20)
    plt.title("Distribucija vjerovatnoća")
    plt.show()


def df_to_matrix(df):
    return df.iloc[:, :].values
