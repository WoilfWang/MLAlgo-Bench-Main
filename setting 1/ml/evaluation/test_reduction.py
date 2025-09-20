import warnings
warnings.filterwarnings("ignore")
import signal
import os
import sys
sys.path.append(os.getcwd())


from sklearn.decomposition import NMF, PCA, KernelPCA, IncrementalPCA, MiniBatchNMF, MiniBatchSparsePCA, SparsePCA, TruncatedSVD
from sklearn.manifold import Isomap, TSNE

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--evaluate_llm', type=str, default='gpt3_5')
parser.add_argument('--result_file', type=str, default='result.csv')
args = parser.parse_args()

exec(f"from llm_answer.{args.evaluate_llm}.reduction import *")

n_components = 2

GPT_MODEL = {
    'nmf': {'model': GPTNMF, 'args': {'n_components': n_components}},
    'isomap': {'model': GPTIsomap, 'args': {'n_components': n_components, 'n_neighbors': 5}},
    'tsne': {'model': GPTTSNE, 'args': {'n_components': n_components, 'perplexity': 30}},
    'pca': {'model': GPTPCA, 'args': {'n_components': n_components}},
    'kernelpca': {'model': GPTKernelPCA, 'args': {'n_components': n_components}},
    'incrementalpca': {'model': GPTIncrementalPCA, 'args': {'n_components': n_components, 'batch_size': None}},
    'mininmf': {'model': GPTMiniBatchNMF, 'args': {'n_components': n_components, 'batch_size': 20}},
    'sparsepca': {'model': GPTSparsePCA, 'args': {'n_components': n_components, 'alpha': 0.5, 'ridge_alpha': 0.01}},
    'minisparsepca': {'model': GPTMiniBatchSparsePCA, 'args': {'n_components': n_components, 'alpha': 0.5, 'ridge_alpha': 0.01, 'batch_size': 20}},
    'truncatedsvd': {'model': GPTTruncatedSVD, 'args': {'n_components': n_components}}
}

SKLEARN_MODEL = {
    'nmf': {'model': NMF, 'args': {'n_components': n_components}},
    'isomap': {'model': Isomap, 'args': {'n_components': n_components, 'n_neighbors': 5}},
    'tsne': {'model': TSNE, 'args': {'n_components': n_components}},
    'pca': {'model': PCA, 'args': {'n_components': n_components}},
    'kernelpca': {'model': KernelPCA, 'args': {'n_components': n_components, 'kernel': 'rbf'}},
    'incrementalpca': {'model': IncrementalPCA, 'args': {'n_components': n_components, 'batch_size': None}},
    'mininmf': {'model': MiniBatchNMF, 'args': {'n_components': n_components, 'batch_size': 20}},
    'sparsepca': {'model': SparsePCA, 'args': {'n_components': n_components, 'alpha': 0.5, 'ridge_alpha': 0.01}},
    'minisparsepca': {'model': MiniBatchSparsePCA, 'args': {'n_components': n_components, 'alpha': 0.5, 'ridge_alpha': 0.01, 'batch_size': 20}},
    'truncatedsvd': {'model': TruncatedSVD, 'args': {'n_components': n_components}}
}

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")

timeout_seconds = 10

results = {'name': [], 'gpt acc': [], 'sklearn acc': [], 'gpt time': [], 'sklearn time': [], 'acc differ': [], 'time differ': []}

for model in GPT_MODEL:
# for model in ['minisparsepca']:
    results['name'].append(model)
    X, y = load_wine(return_X_y=True)
    try:
        gptmodel = GPT_MODEL[model]['model'](**GPT_MODEL[model]['args'])
        t1 = time.time()
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            X_reduction = gptmodel.fit_transform(X)
            if X_reduction.shape[1] != n_components:
                raise ValueError('reduction error')
            signal.alarm(0)
        except TimeoutException as e:
            print(e)
        finally:
            signal.alarm(0)
        
        X_train, X_test, y_train, y_test = train_test_split(X_reduction, y, test_size=0.3, random_state=42)
        t2 = time.time()
        
        class_model = LogisticRegression(random_state=42)
        class_model.fit(X_train, y_train)
        predictions = class_model.predict(X_test)
        
        gpt_acc = accuracy_score(predictions, y_test)
        gpt_time = t2 - t1
        
    except Exception as e:
        print(f"gpt model {model} error,", e)
        gpt_acc = -1
    
    if model in SKLEARN_MODEL:
        X, y = load_wine(return_X_y=True)
        try:
            sklearn_model = SKLEARN_MODEL[model]['model'](**SKLEARN_MODEL[model]['args'])
            t1 = time.time()

            X_reduction = sklearn_model.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_reduction, y, test_size=0.3, random_state=42)
            t2 = time.time()
            
            class_model = LogisticRegression(random_state=42)
            class_model.fit(X_train, y_train)
            predictions = class_model.predict(X_test)
            results['sklearn acc'].append(accuracy_score(predictions, y_test))
            results['sklearn time'].append(t2 - t1)
            
            sklearn_acc = accuracy_score(predictions, y_test)
            sklearn_time = t2 - t1
            
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