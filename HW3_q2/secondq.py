from surprise import Dataset, Reader, model_selection
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

# Store the results
results_dict = {'Algorithm': [], 'MAE': [], 'RMSE': []}

# Perform 5-fold cross-validation for each algorithm
for algo in [user_cf, item_cf, pmf]:
    for metric in metrics:
        results = model_selection.cross_validate(algo, data, measures=[metric], cv=5, verbose=False)
        avg_metric = sum(results[f'test_{metric.lower()}']) / 5
        results_dict['Algorithm'].append(algo.__class__.__name__)
        results_dict[metric].append(avg_metric)

# Print the results
print("\nResults:")
for metric in metrics:
    print(f"\nAverage {metric} across algorithms:")
    for algo, value in zip(results_dict['Algorithm'], results_dict[metric]):
        print(f"{algo}: {value}")
