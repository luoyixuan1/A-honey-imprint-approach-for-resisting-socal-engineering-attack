from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from time import time
import pickle
from sklearnex import patch_sklearn, config_context


def SVM(matrix_train, label_train, matrix_test, label_test):
    size = 100
    epoch = len(matrix_train) // size
    accuracy_ls = []
    precision_ls = []
    recall_ls = []
    f1_score_ls = []
    auc_ls = []

    for i in range(epoch):
        print(f'{i} / {epoch}')
        svm_clf = svm.SVC(gamma='scale')
        svm_clf.fit(matrix_train[:(i+1)*size, :], label_train[:(i+1)*size])

        predictions_svm = svm_clf.predict(matrix_test)
        probs_svm = svm_clf.predict_proba(matrix_test)[:, 1]

        accuracy_ls.append(accuracy_score(label_test, predictions_svm))
        precision_ls.append(precision_score(label_test, predictions_svm))
        recall_ls.append(recall_score(label_test, predictions_svm))
        f1_score_ls.append(f1_score(label_test, predictions_svm))
        auc_ls.append(roc_auc_score(label_test, probs_svm))

    print(f'{epoch} / {epoch}')
    start = time()
    svm_clf = svm.SVC(gamma='scale')
    svm_clf.fit(matrix_train, label_train)
    end = time()

    predictions_svm = svm_clf.predict(matrix_test)
    probs_svm = svm_clf.predict_proba(matrix_test)[:, 1]

    accuracy_ls.append(accuracy_score(label_test, predictions_svm))
    precision_ls.append(precision_score(label_test, predictions_svm))
    recall_ls.append(recall_score(label_test, predictions_svm))
    f1_score_ls.append(f1_score(label_test, predictions_svm))
    auc_ls.append(roc_auc_score(label_test, probs_svm))

    return predictions_svm, [accuracy_ls, precision_ls, recall_ls, f1_score_ls], end-start


def RF(matrix_train, label_train, matrix_test, label_test):
    size = 100
    epoch = len(matrix_train) // size
    accuracy_ls = []
    precision_ls = []
    recall_ls = []
    f1_score_ls = []
    auc_ls = []

    for i in range(epoch):
        print(f'{i} / {epoch}')
        random_forest = RandomForestClassifier()
        random_forest.fit(matrix_train[:(i+1)*size, :], label_train[:(i+1)*size])

        predictions_rf = random_forest.predict(matrix_test)
        probs_rf = random_forest.predict_proba(matrix_test)[:, 1]

        accuracy_ls.append(accuracy_score(label_test, predictions_rf))
        precision_ls.append(precision_score(label_test, predictions_rf))
        recall_ls.append(recall_score(label_test, predictions_rf))
        f1_score_ls.append(f1_score(label_test, predictions_rf))
        auc_ls.append(roc_auc_score(label_test, probs_rf))

    print(f'{epoch} / {epoch}')
    start = time()
    random_forest = RandomForestClassifier()
    random_forest.fit(matrix_train, label_train)
    end = time()

    predictions_rf = random_forest.predict(matrix_test)
    probs_rf = random_forest.predict_proba(matrix_test)[:, 1]

    accuracy_ls.append(accuracy_score(label_test, predictions_rf))
    precision_ls.append(precision_score(label_test, predictions_rf))
    recall_ls.append(recall_score(label_test, predictions_rf))
    f1_score_ls.append(f1_score(label_test, predictions_rf))
    auc_ls.append(roc_auc_score(label_test, probs_rf))

    return predictions_rf, [accuracy_ls, precision_ls, recall_ls, f1_score_ls, auc_ls], end-start


def logistic(matrix_train, label_train, matrix_test, label_test):
    size = 100
    epoch = len(matrix_train) // size
    accuracy_ls = []
    precision_ls = []
    recall_ls = []
    f1_score_ls = []
    auc_ls = []

    for i in range(epoch):
        print(f'{i} / {epoch}')

        logistic_regression = LogisticRegression()
        logistic_regression.fit(matrix_train[:(i+1)*size, :], label_train[:(i+1)*size])

        predictions_lg = logistic_regression.predict(matrix_test)
        probs_lg = logistic_regression.predict_proba(matrix_test)[:, 1]

        accuracy_ls.append(accuracy_score(label_test, predictions_lg))
        precision_ls.append(precision_score(label_test, predictions_lg))
        recall_ls.append(recall_score(label_test, predictions_lg))
        f1_score_ls.append(f1_score(label_test, predictions_lg))
        auc_ls.append(roc_auc_score(label_test, probs_lg))

    print(f'{epoch} / {epoch}')
    start = time()
    logistic_regression = LogisticRegression()
    logistic_regression.fit(matrix_train, label_train)
    end = time()

    predictions_lg = logistic_regression.predict(matrix_test)
    probs_lg = logistic_regression.predict_proba(matrix_test)[:, 1]

    accuracy_ls.append(accuracy_score(label_test, predictions_lg))
    precision_ls.append(precision_score(label_test, predictions_lg))
    recall_ls.append(recall_score(label_test, predictions_lg))
    f1_score_ls.append(f1_score(label_test, predictions_lg))
    auc_ls.append(roc_auc_score(label_test, probs_lg))

    return predictions_lg, [accuracy_ls, precision_ls, recall_ls, f1_score_ls, auc_ls], end-start


def MLP(matrix_train, label_train, matrix_test, label_test):
    size = 100
    epoch = len(matrix_train) // size
    accuracy_ls = []
    precision_ls = []
    recall_ls = []
    f1_score_ls = []

    for i in range(epoch):
        print(f'{i} / {epoch}')

        mlp_clf = MLPClassifier(random_state=1)
        mlp_clf.partial_fit(matrix_train[:(i+1)*size, :], label_train[:(i+1)*size], classes=np.array([0, 1]))
        predictions_mlp = mlp_clf.predict(matrix_test)

        accuracy_ls.append(accuracy_score(label_test, predictions_mlp))
        precision_ls.append(precision_score(label_test, predictions_mlp))
        recall_ls.append(recall_score(label_test, predictions_mlp))
        f1_score_ls.append(f1_score(label_test, predictions_mlp))

    print(f'{epoch} / {epoch}')
    start = time()
    mlp_clf = MLPClassifier(random_state=1)
    mlp_clf.partial_fit(matrix_train, label_train, classes=np.array([0, 1]))
    end = time()

    predictions_mlp = mlp_clf.predict(matrix_test)

    accuracy_ls.append(accuracy_score(label_test, predictions_mlp))
    precision_ls.append(precision_score(label_test, predictions_mlp))
    recall_ls.append(recall_score(label_test, predictions_mlp))
    f1_score_ls.append(f1_score(label_test, predictions_mlp))

    return predictions_mlp, [accuracy_ls, precision_ls, recall_ls, f1_score_ls], end-start


def module_evaluate(name, true_label, predict_label, time_span):
    with open('result.csv', 'a') as file:
        print('{} Accuracy score: {}'.format(name, accuracy_score(true_label, predict_label)))
        print('{} Precision score: {}'.format(name, precision_score(true_label, predict_label)))
        print('{} Recall score: {}'.format(name, recall_score(true_label, predict_label)))
        print('{} F1 score: {}'.format(name, f1_score(true_label, predict_label)))
        tn, fp, fn, tp = confusion_matrix(true_label, predict_label).ravel()
        file.write('{},{:.5f},{:.5f},{:.5f},{:.5f},,{},{},{},{},, {}\n'.format(name,
                                                             accuracy_score(true_label, predict_label),
                                                             precision_score(true_label, predict_label),
                                                             recall_score(true_label, predict_label),
                                                             f1_score(true_label, predict_label),
                                                            tp, tn, fp, fn, time_span))

        print(len(predict_label))
        print(confusion_matrix(true_label, predict_label))


def main():
    patch_sklearn()
    experiment_name = 'mlp_dataset_wordFrequency_sentiment_normalization---'

    file = pd.read_csv('dataset.csv', header=0, encoding_errors='ignore')
    count_vector = CountVectorizer(stop_words='english')

    data = count_vector.fit_transform(file['text'], )
    matrix = data.toarray()
    # matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    matrix = matrix.astype(np.float32)
    print('===word frequency analysis complete===')

    polarity = file['polarity']
    subjectivity = file['subjectivity']
    matrix = np.insert(matrix, 0, polarity, axis=1)
    matrix = np.insert(matrix, 0, subjectivity, axis=1)
    print('===sentiment analysis complete===')

    x_train, x_test, y_train, y_test = train_test_split(matrix, file['label_num'], random_state=1)

    words_train = x_train[:, 2:]
    words_test = x_test[:, 2:]

    sentiment_train = x_train[:, :2]
    sentiment_test = x_test[:, :2]

    predictions_svm, evaluation_parameter, time_span = SVM(x_train, y_train, x_test, y_test)
    # predictions_svm, evaluation_parameter, time_span = SVM(words_train, y_train, words_test, y_test)
    # predictions_svm, evaluation_parameter, time_span = SVM(sentiment_train, y_train, sentiment_test, y_test)

    predictions_rf, evaluation_parameter, time_span = RF(x_train, y_train, x_test, y_test)
    # predictions_rf, evaluation_parameter, time_span = RF(words_train, y_train, words_test, y_test)
    # predictions_rf, evaluation_parameter, time_span = RF(sentiment_train, y_train, sentiment_test, y_test)

    predictions_lg, evaluation_parameter, time_span = logistic(x_train, y_train, x_test, y_test)
    # predictions_lg, evaluation_parameter, time_span = logistic(words_train, y_train, words_test, y_test)
    # predictions_lg, evaluation_parameter, time_span = logistic(sentiment_train, y_train, sentiment_test, y_test)

    predictions_mlp, evaluation_parameter, time_span = MLP(x_train, y_train, x_test, y_test)
    # predictions_mlp, evaluation_parameter, time_span = MLP(words_train, y_train, words_test, y_test)
    # predictions_mlp, evaluation_parameter, time_span = MLP(sentiment_train, y_train, sentiment_test, y_test)

    print('===training complete===')
    # 模型评估
    module_evaluate(experiment_name, y_test, predictions_svm, time_span)
    module_evaluate(experiment_name, y_test, predictions_rf, time_span)
    module_evaluate(experiment_name, y_test, predictions_lg, time_span)
    module_evaluate(experiment_name, y_test, predictions_mlp, time_span)

    with open(f'{experiment_name}.data', 'wb') as file:
        pickle.dump(evaluation_parameter, file)
        

if __name__ == '__main__':
    main()
