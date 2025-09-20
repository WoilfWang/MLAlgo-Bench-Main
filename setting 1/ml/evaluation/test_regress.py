import warnings
warnings.filterwarnings("ignore")
import signal
import os
import sys
sys.path.append(os.getcwd())


from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import (
    AdaBoostRegressor, 
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    BaggingRegressor
)
from sklearn.neighbors import (
    RadiusNeighborsRegressor,
    KNeighborsRegressor
)
from sklearn.linear_model import (
    Lasso, 
    Ridge,
    LinearRegression,
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    GammaRegressor,
    PoissonRegressor,
    TweedieRegressor,
    HuberRegressor,
    Lars,
    LassoLars,
    PassiveAggressiveRegressor,
    SGDRegressor,
    RANSACRegressor,
    TheilSenRegressor
)

from sklearn.neural_network import MLPRegressor


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--evaluate_llm', type=str, default='gpt3_5')
parser.add_argument('--result_file', type=str, default='result.csv')
args = parser.parse_args()

exec(f"from llm_answer.{args.evaluate_llm}.regression import *")

GPT_MODEL = {
    'decision_tree': {'model': GPTDecisionRegressionTree, 'args': {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}},
    'mlp': {'model': GPTMLP, 'args': {'hidden_layer_sizes': [52, 1], 'alpha': 0.0001, 'batch_size': 32, 'learning_rate_init': 1e-3, 'max_iter': 50}},
    'adaboost': {'model': GPTAdaboostRegressor, 'args': {'n_estimators': 50, 'learning_rate': 1.0}},
    'random_forest': {'model': GPTRandomForestRegressor, 'args': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}},
    'lasso': {'model': GPTLassoRegression, 'args': {'alpha': 0.1}},
    'ridge': {'model': GPTRidgeRegression, 'args': {'alpha': 0.1}},
    'gbdt': {'model': GPTGradientBoostRegression, 'args': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 1.0}},
    'extretree': {'model': GPTExtraTreeRegressor, 'args': {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}},
    'radius_knn': {'model': GPTRadiusNeighborsRegressor, 'args': {"radius": 4.0}},
    'linear': {'model': GPTLinearRegressor, 'args': {}},
    'ard': {'model': GPTARDRegressor, 'args': {'max_iter': 300, 'alpha_1': 1e-6, 'alpha_2': 1e-6, 'lambda_1': 1e-6, 'lambda_2': 1e-6}},
    'bayes_ridge': {'model': GPTBayesRidgeRegressor, 'args': {'max_iter': 300, 'alpha_1': 1e-6, 'alpha_2': 1e-6, 'lambda_1': 1e-6, 'lambda_2': 1e-6}},
    'elastic': {'model': GPTElasticNet, 'args': {'alpha': 1.0, 'l1_ratio': 0.5}},
    'gamma': {'model': GPTGammaRegressor, 'args': {'alpha': 0.5}},
    'poisson': {'model': GPTPoissonRegressor, 'args': {'alpha': 0.5}},
    'tweedie': {'model': GPTTweedieRegressor, 'args': {'alpha': 0.5, 'power': 0.0}},
    'huber': {'model': GPTHuberRegressor, 'args': {'epsilon': 1.35, 'max_iter': 100, 'alpha': 0.0001}},
    'hgbdt': {'model': GPTHistGradientBoostingRegressor, 'args': {'learning_rate': 0.1, 'max_iter': 100, 'max_leaf_nodes': 31, 'max_depth': None, 'min_samples_leaf': 20}},
    'bagging': {'model': GPTBagging, 'args': {'n_estimators': 10, 'max_samples': 1.0, 'max_features': 1.0}},
    'lars': {'model': GPTLars, 'args': {}},
    'lassolars': {'model': GPTLassoLars, 'args': {'alpha': 1.0}},
    'passive_aggressive': {'model': GPTPassiveAggressive, 'args': {'C': 1.0}},
    'sgd_svm': {'model': GPTSGDSVM, 'args': {}},
    'sanrac': {'model': GPTRANSACRegressor, 'args': {'estimator': LinearRegression()}},
    'theilsen': {'model': GPTTheilSenRegressor, 'args': {}},
    'knn': {'model': GPTKNN, 'args': {'n_neighbors': 5}}
}

SKLEARN_MODEL = {
    'decision_tree': {'model': DecisionTreeRegressor, 'args': {}},
    'mlp': {'model': MLPRegressor, 'args': {'hidden_layer_sizes': [52], 'alpha': 0.0001, 'batch_size': 32, 'learning_rate_init': 1e-3, 'max_iter': 50}},
    'bayes': {'model': BayesianRidge, 'args': {}},
    'adaboost': {'model': AdaBoostRegressor, 'args': {'n_estimators': 50}},
    'random_forest': {'model': RandomForestRegressor, 'args': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}},
    'lasso': {'model': Lasso, 'args': {'alpha': 0.1}},
    'ridge': {'model': Ridge, 'args': {'alpha': 0.1}},
    'gbdt': {'model': GradientBoostingRegressor, 'args': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1}},
    'extretree': {'model': ExtraTreesRegressor, 'args': {}},
    'radius_knn': {'model': RadiusNeighborsRegressor, 'args': {"radius": 4.0}},
    'linear': {'model': LinearRegression, 'args': {}},
    'ard': {'model': ARDRegression, 'args': {'max_iter': 300, 'alpha_1': 1e-6, 'alpha_2': 1e-6, 'lambda_1': 1e-6, 'lambda_2': 1e-6}},
    'bayes_ridge': {'model': BayesianRidge, 'args': {'max_iter': 300, 'alpha_1': 1e-6, 'alpha_2': 1e-6, 'lambda_1': 1e-6, 'lambda_2': 1e-6}},
    'elastic': {'model': ElasticNet, 'args': {'alpha': 1.0, 'l1_ratio': 0.5}},
    'gamma': {'model': GammaRegressor, 'args': {'alpha': 0.5}},
    'poisson': {'model': PoissonRegressor, 'args': {'alpha': 0.5}},
    'tweedie': {'model': TweedieRegressor, 'args': {'alpha': 0.5, 'power': 0.0}},
    'huber': {'model': HuberRegressor, 'args': {}},
    'hgbdt': {'model': HistGradientBoostingRegressor, 'args': {}},
    'bagging': {'model': BaggingRegressor, 'args': {}},
    'lars': {'model': Lars, 'args': {}},
    'lassolars': {'model': LassoLars, 'args': {}},
    'passive_aggressive': {'model': PassiveAggressiveRegressor, 'args': {}},
    'sgd_svm': {'model': SGDRegressor, 'args': {'loss': 'epsilon_insensitive'}},
    'sanrac': {'model': RANSACRegressor, 'args': {}},
    'theilsen': {'model': TheilSenRegressor, 'args': {}},
    'knn': {'model': KNeighborsRegressor, 'args': {'n_neighbors': 5}}
}


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd
from tqdm import tqdm

def load_boston_house_price():
    data = pd.read_csv('evaluation-data/housing.csv')
    X = data.drop(['MEDV'], axis=1).values
    y = data['MEDV'].values
    return X, y

results = {'name': [], 'gpt mse': [], 'sklearn mse': [], 'gpt time': [], 'sklearn time': [], 'mse differ': [], 'time differ': []}

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")

timeout_seconds = 10

for model in tqdm(GPT_MODEL):
    results['name'].append(model)
    X, y = load_boston_house_price()
    standard = StandardScaler()
    X = standard.fit_transform(X)
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
        
        gpt_mse = mean_squared_error(predictions, y_test)
        gpt_time = t2 - t1
        
        
    except Exception as e:
        print(f"gpt model {model} error,", e)
        gpt_mse = 1e7
    
    if model in SKLEARN_MODEL:
        X, y = load_boston_house_price()
        standard = StandardScaler()
        X = standard.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        try:
            sklearn_model = SKLEARN_MODEL[model]['model'](**SKLEARN_MODEL[model]['args'])
            t1 = time.time()
            sklearn_model.fit(X_train, y_train)
            predictions = sklearn_model.predict(X_test)
            t2 = time.time()
            results['sklearn mse'].append(mean_squared_error(predictions, y_test))
            results['sklearn time'].append(t2 - t1)
            sklearn_mse = mean_squared_error(predictions, y_test)
            sklearn_time = t2 - t1
        except Exception as e:
            print(f"sklearn model {model} error,", e)
            results['sklearn mse'].append('failed')
            results['sklearn time'].append('failed')
    else:
        results['sklearn mse'].append('')
        results['sklearn time'].append('')
        
    if gpt_mse > sklearn_mse + 60:
        results['gpt mse'].append('failed')
        results['gpt time'].append('failed')
    else:
        results['gpt mse'].append(gpt_mse)
        results['gpt time'].append(gpt_time)
        
    if results['gpt mse'][-1] != 'failed' and results['sklearn mse'][-1] != 'failed':
        results['mse differ'].append(results['gpt mse'][-1] - results['sklearn mse'][-1])
        results['time differ'].append(results['gpt time'][-1] - results['sklearn time'][-1])
    else:
        results['mse differ'].append('failed')
        results['time differ'].append('failed')

results = pd.DataFrame(results)
results.to_csv(args.result_file, index=None)

def mean_of_floats(column):
    column_numeric = pd.to_numeric(column, errors='coerce')
    return column_numeric.mean()

mean_values = results.apply(mean_of_floats)
print(mean_values)

pass_rate = results['gpt mse'].apply(lambda x: isinstance(x, float)).sum() / results.shape[0]
print('pass rate: ', pass_rate)