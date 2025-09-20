import warnings
warnings.filterwarnings("ignore")
import signal
import os
import sys
sys.path.append(os.getcwd())

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans, SpectralClustering, SpectralBiclustering, SpectralCoclustering, BisectingKMeans
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--evaluate_llm', type=str, default='gpt3_5')
parser.add_argument('--result_file', type=str, default='result.csv')
args = parser.parse_args()

exec(f"from llm_answer.{args.evaluate_llm}.cluster import *")

n_clusters = 3

GPT_MODEL = {
    'kmeans': {'model': GPTKmeans, 'args': {'n_clusters': n_clusters}},
    'gm': {'model': GPTGaussianMixture, 'args': {'n_components': n_clusters}},
    'dbscan': {'model': GPTDBSCAN, 'args': {'eps': 0.5, 'min_samples': 5}},
    'hierarchical': {'model': GPTHierarchicalCluster, 'args': {'n_clusters': n_clusters}},
    'hdbscan': {'model': GPTHDBSCAN, 'args': {'min_cluster_size': 5}},
    'minikmeans': {'model': GPTMinibatchKmeans, 'args': {'n_clusters': n_clusters, 'batch_size': 32}},
    'spectral': {'model': GPTSpectralClustering, 'args': {'n_clusters': n_clusters}},
    'biclustering': {'model': GPTSpectralBiClustering, 'args': {'n_clusters': n_clusters}},
    'coclustering': {'model': GPTSpectralCoClustering, 'args': {'n_clusters': n_clusters}},
    'bisecting': {'model': GPTBisectingKmeans, 'args': {'n_clusters': n_clusters}}
}

SKLEARN_MODEL = {
    'kmeans': {'model': KMeans, 'args': {'n_clusters': n_clusters}},
    'gm': {'model': GaussianMixture, 'args': {'n_components': n_clusters}},
    'dbscan': {'model': DBSCAN, 'args': {'eps': 0.5, 'min_samples': 5}},
    'hierarchical': {'model': AgglomerativeClustering, 'args': {'n_clusters': n_clusters}},
    'hdbscan': {'model': HDBSCAN, 'args': {'min_cluster_size': 5}},
    'minikmeans': {'model': MiniBatchKMeans, 'args': {'n_clusters': n_clusters, 'batch_size': 32}},
    'spectral': {'model': SpectralClustering, 'args': {'n_clusters': n_clusters}},
    'biclustering': {'model': SpectralBiclustering, 'args': {'n_clusters': n_clusters}},
    'coclustering': {'model': SpectralCoclustering, 'args': {'n_clusters': n_clusters}},
    'bisecting': {'model': BisectingKMeans, 'args': {'n_clusters': n_clusters}}
}


from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd

results = {'name': [], 'gpt score': [], 'sklearn score': [], 'gpt time': [], 'sklearn time': [], 'score differ': [], 'time differ': []}

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超时")

timeout_seconds = 10

for model in tqdm(GPT_MODEL):
    results['name'].append(model)
    X, _ = load_iris(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    try:
        gptmodel = GPT_MODEL[model]['model'](**GPT_MODEL[model]['args'])
        t1 = time.time()
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            predictions = gptmodel.fit_predict(X)
            signal.alarm(0)
        except TimeoutException as e:
            print(e)
        finally:
            signal.alarm(0)
            
        t2 = time.time()
        gpt_score = silhouette_score(X, predictions)
        gpt_time = t2 - t1
        
    except Exception as e:
        print(f"gpt model {model} error,", e)
        gpt_score = -1
    
    if model in SKLEARN_MODEL:
        X, _ = load_iris(return_X_y=True)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        try:
            sklearn_model = SKLEARN_MODEL[model]['model'](**SKLEARN_MODEL[model]['args'])
            t1 = time.time()
            if model in ['biclustering', 'coclustering']:
                sklearn_model.fit(X)
                predictions = sklearn_model.row_labels_
            else:
                predictions = sklearn_model.fit_predict(X)
            t2 = time.time()
            results['sklearn score'].append(silhouette_score(X, predictions))
            results['sklearn time'].append(t2 - t1)
            sklearn_score = silhouette_score(X, predictions)
            sklearn_time = t2 - t1
        except Exception as e:
            print(f"sklearn model {model} error,", e)
            results['sklearn score'].append('failed')
            results['sklearn time'].append('failed')
    else:
        results['sklearn score'].append('')
        results['sklearn time'].append('')
        
    if gpt_score < sklearn_score - 0.1:
        results['gpt score'].append('failed')
        results['gpt time'].append('failed')
    else:
        results['gpt score'].append(gpt_score)
        results['gpt time'].append(gpt_time)   
        
    if results['gpt score'][-1] != 'failed' and results['sklearn score'][-1] != 'failed':
        results['score differ'].append(results['gpt score'][-1] - results['sklearn score'][-1])
        results['time differ'].append(results['gpt time'][-1] - results['sklearn time'][-1])
    else:
        results['score differ'].append('failed')
        results['time differ'].append('failed')

results = pd.DataFrame(results)
results.to_csv(args.result_file, index=None)

def mean_of_floats(column):
    column_numeric = pd.to_numeric(column, errors='coerce')
    return column_numeric.mean()

mean_values = results.apply(mean_of_floats)
print(mean_values)

pass_rate = results['gpt score'].apply(lambda x: isinstance(x, float)).sum() / results.shape[0]
print('pass rate: ', pass_rate)
    

