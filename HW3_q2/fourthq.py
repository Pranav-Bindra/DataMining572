import matplotlib.pyplot as plt
import numpy as np
from surprise import Dataset, Reader, model_selection
from surprise.prediction_algorithms.knns import KNNBasic

file_path = 'archive/ratings_small.csv'

reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(file_path, reader)

# Define the range of neighbors to be tested
neighbors_range = [5, 10, 15, 20, 25]

# Store the results
results_dict = {'Neighbors': [], 'MAE': [], 'RMSE': []}

for n_neighbors in neighbors_range:
    # Define the algorithms
    user_cf = KNNBasic(k=n_neighbors, sim_options={'user_based': True})
    item_cf = KNNBasic(k=n_neighbors, sim_options={'user_based': False})

    # Perform 5-fold cross-validation for each number of neighbors
    for algo, name in zip([user_cf, item_cf], ['User-based CF', 'Item-based CF']):
        results = model_selection.cross_validate(algo, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
        avg_mae = np.mean(results['test_mae'])
        avg_rmse = np.mean(results['test_rmse'])
        results_dict['Neighbors'].append(f'{name} ({n_neighbors})')
        results_dict['MAE'].append(avg_mae)
        results_dict['RMSE'].append(avg_rmse)

# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Bar graph for MAE
ax[0].bar(results_dict['Neighbors'], results_dict['MAE'], color='blue', alpha=0.7)
ax[0].set_title('MAE')
ax[0].set_xlabel('Number of Neighbors')
ax[0].set_ylabel('Performance')

# Bar graph for RMSE
ax[1].bar(results_dict['Neighbors'], results_dict['RMSE'], color='orange', alpha=0.7)
ax[1].set_title('RMSE')
ax[1].set_xlabel('Number of Neighbors')
ax[1].set_ylabel('Performance')

plt.suptitle('Impact of Number of Neighbors on Collaborative Filtering')

# Rotate x-axis labels for both subplots
for a in ax:
    a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent title overlap
plt.show()
