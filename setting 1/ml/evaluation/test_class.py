import warnings
warnings.filterwarnings("ignore")
import signal
import os
import sys
sys.path.append(os.getcwd())


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    SGDClassifier,
    RidgeClassifier,
    PassiveAggressiveClassifier
)
import traceback
from sklearn.neural_network import MLPClassifier    
from xgboost import XGBClassifier


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--evaluate_llm', type=str, default='gpt3_5')
parser.add_argument('--result_file', type=str, default='result.csv')
args = parser.parse_args()

exec(f"from llm_answer.{args.evaluate_llm}.classification import *")


num_classes = 10

GPT_MODEL = {
    'lc': {'model': GPTLogisticRegression, 'args': {'num_classes': num_classes}},
    'decisiontree': {'model': GPTDecisionClassificationTree, 'args': {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}},
    'svm': {'model': GPTSVM, 'args': {'C': 1.0}},
    'gbdt': {'model': GPTGradientBoostDecisionTree, 'args': {'n_estimators': 100, 'learning_rate': 0.1, 'subsample': 1.0, 'max_depth': 3}},
    'lda': {'model': GPTLinearDiscrimination, 'args': {'num_classes': num_classes}},
    'bayes': {'model': GPTNaiveBayesClassifier, 'args': {}},
    'adaboost': {'model': GPTAdaboostClassifier, 'args': {'n_estimators': 50, 'learning_rate': 1.0}},
    'randomforest': {'model': GPTRandomForestClassifier, 'args': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}},
    'knn': {'model': GPTKNN, 'args': {'n_neighbors': 3}},
    'xgboost': {'model': GPTXGboostClassifier, 'args': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}},
    'mlp': {'model': GPTMLP, 'args': {'hidden_layer_sizes': [52, num_classes], 'alpha': 0.0001, 'batch_size': 32, 'learning_rate_init': 1e-3}},
    'extratree': {'model': GPTExtraTreesClassifier, 'args': {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}},
    'radius_knn': {'model': GPTRadiusNeighborsClassifier, 'args': {'radius': 10.0}},
    'hgbdt': {'model': GPTHistGradientBoostingClassifier, 'args': {'learning_rate': 0.1, 'max_iter': 100, 'max_leaf_nodes': 31, 'max_depth': None, 'min_samples_leaf': 20}},
    'sgd_svm': {'model': GPTSgdSvm, 'args': {}},
    'gaussian_process': {'model': GPTGPC, 'args': {}},
    'sgd_lr': {'model': GPTSgdLR, 'args': {}},
    'ridge': {'model': GPTRidge, 'args': {'alpha': 0.1}},
    'passive_aggressive': {'model': GPTPassiveAggressive, 'args': {'C': 1.0}},
    'bagging': {'model': GPTBagging, 'args': {'n_estimators': 10, 'max_samples': 1.0, 'max_features': 1.0}},
}

SKLEARN_MODEL = {
    'lc': {'model': LogisticRegression, 'args': {}},
    'decisiontree': {'model': DecisionTreeClassifier, 'args': {}},
    'svm': {'model': SVC, 'args':{'kernel': 'rbf', 'C': 1.0}},
    'gbdt': {'model': GradientBoostingClassifier, 'args': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}},
    'lda': {'model': LinearDiscriminantAnalysis, 'args': {}},
    'bayes': {'model': GaussianNB, 'args': {}},
    'adaboost': {'model': AdaBoostClassifier, 'args': {'n_estimators': 50, 'learning_rate': 1.0, 'estimator': DecisionTreeClassifier(max_depth=3)}},
    'randomforest': {'model': RandomForestClassifier, 'args': {'n_estimators': 100, 'max_depth': None}},
    'knn': {'model': KNeighborsClassifier, 'args': {'n_neighbors': 3}},
    'xgboost': {'model': XGBClassifier, 'args': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}},
    'mlp': {'model': MLPClassifier, 'args': {'hidden_layer_sizes': [52, num_classes], 'alpha': 0.0001, 'batch_size': 32, 'learning_rate_init': 1e-3}},
    'extratree': {'model': ExtraTreesClassifier, 'args': {}},
    'radius_knn': {'model': RadiusNeighborsClassifier, 'args': {'radius': 10.0}},
    'hgbdt': {'model': HistGradientBoostingClassifier, 'args': {}},
    'sgd_svm': {'model': SGDClassifier, 'args': {'loss': 'hinge'}},
    'gaussian_process': {'model': GaussianProcessClassifier, 'args': {'n_jobs': -1}},
    'sgd_lr': {'model': SGDClassifier, 'args': {'loss': 'log_loss'}},
    'ridge': {'model': RidgeClassifier, 'args': {'alpha': 0.1}},
    'passive_aggressive': {'model': PassiveAggressiveClassifier, 'args': {'C': 1.0}},
    'bagging': {'model': BaggingClassifier, 'args': {'n_estimators': 10, 'max_samples': 1.0, 'max_features': 1.0}},
}



from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd
from sklearn.decomposition import PCA

results = {'name': [], 'gpt acc': [], 'sklearn acc': [], 'gpt time': [], 'sklearn time': [], 'acc differ': [], 'time differ': []}


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")

timeout_seconds = 10

for model in GPT_MODEL:
    results['name'].append(model)
    X, y = load_digits(return_X_y=True)
    standarder = StandardScaler()
    X = standarder.fit_transform(X)
    pca = PCA(n_components=24)
    pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    try:
        gptmodel = GPT_MODEL[model]['model'](**GPT_MODEL[model]['args'])
        t1 = time.time()
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            gptmodel.fit(X_train, y_train)
            predictions = gptmodel.predict(X_test)
            signal.alarm(0)
        except TimeoutException as e:
            print(e)
        finally:
            signal.alarm(0)
        
        t2 = time.time()
        
        gpt_acc = accuracy_score(predictions, y_test)
        gpt_time = t2 - t1
        
    except Exception as e:
        print(f"gpt model {model} error,", e)
        gpt_acc = -1
        traceback.print_exc()
    
    if model in SKLEARN_MODEL:
        X, y = load_digits(return_X_y=True)
        standarder = StandardScaler()
        X = standarder.fit_transform(X)
        pca = PCA(n_components=24)
        pca.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        try:
            sklearn_model = SKLEARN_MODEL[model]['model'](**SKLEARN_MODEL[model]['args'])
            t1 = time.time()
            sklearn_model.fit(X_train, y_train)
            predictions = sklearn_model.predict(X_test)
            t2 = time.time()
            
            sklearn_acc = accuracy_score(predictions, y_test)
            sklearn_time = t2 - t1
            
            results['sklearn acc'].append(sklearn_acc)
            results['sklearn time'].append(sklearn_time) 
            
        except Exception as e:
            print(f"sklearn model {model} error,", e)
            results['sklearn acc'].append('failed')
            results['sklearn time'].append('failed')
    else:
        results['sklearn acc'].append('')
        results['sklearn time'].append('')
        
    if gpt_acc < sklearn_acc - 0.2:
        results['gpt acc'].append('failed')
        results['gpt time'].append('failed')
    else:
        results['gpt acc'].append(gpt_acc)
        results['gpt time'].append(gpt_time)   
        
    if results['gpt acc'][-1] != 'failed' and results['sklearn acc'][-1] != 'failed':
        results['acc differ'].append(results['gpt acc'][-1] - results['sklearn acc'][-1])
        results['time differ'].append(results['gpt time'][-1] - results['sklearn time'][-1])
    else:
        results['acc differ'].append('failed')
        results['time differ'].append('failed')
    

results = pd.DataFrame(results)
results.to_csv(args.result_file, index=None)

def mean_of_floats(column):
    column_numeric = pd.to_numeric(column, errors='coerce')
    return column_numeric.mean()

mean_values = results.apply(mean_of_floats)
print(mean_values)

pass_rate = results['gpt acc'].apply(lambda x: isinstance(x, float)).sum() / results.shape[0]
print('pass rate: ', pass_rate)
    
    
    