from surprise import Dataset, Reader, model_selection, accuracy
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.knns import KNNBasic

# Load data from the CSV file
file_path = 'archive/ratings_small.csv'

reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(file_path, reader)

# Define the algorithms
pmf = SVD(biased=False)  # Probabilistic Matrix Factorization
user_cf = KNNBasic(sim_options={'user_based': True})  # User-based Collaborative Filtering
item_cf = KNNBasic(sim_options={'user_based': False})  # Item-based Collaborative Filtering

# Define the evaluation metrics
metrics = ['MAE', 'RMSE']
for algo in [pmf, user_cf, item_cf]:
    print(f'\nAlgorithm: {algo.__class__.__name__}')
    for metric in metrics:
        results = model_selection.cross_validate(algo, data, measures=[metric], cv=5, verbose=True)
        avg_metric = sum(results[f'test_{metric.lower()}']) / 5  # Use lowercase key
        print(f'Average {metric} across 5 folds: {avg_metric}')