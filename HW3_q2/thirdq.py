import matplotlib.pyplot as plt
import numpy as np
from surprise import Dataset, Reader, model_selection
from surprise.prediction_algorithms.knns import KNNBasic

file_path = 'archive/ratings_small.csv'

reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(file_path, reader)

# Define the algorithms
user_cf = KNNBasic(sim_options={'user_based': True})
item_cf = KNNBasic(sim_options={'user_based': False})

# Define similarity metrics to be tested
similarity_metrics = ['cosine', 'msd', 'pearson']

# Store the results for both User-based and Item-based CF
results_dict_user = {'Metric': [], 'MAE': [], 'RMSE': []}
results_dict_item = {'Metric': [], 'MAE': [], 'RMSE': []}

# Perform 5-fold cross-validation for each similarity metric for both User-based and Item-based CF
for metric in similarity_metrics:
    # User-based CF
    user_cf.sim_options['sim_options'] = {'name': metric}
    results_user = model_selection.cross_validate(user_cf, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
    avg_mae_user = np.mean(results_user['test_mae'])
    avg_rmse_user = np.mean(results_user['test_rmse'])
    results_dict_user['Metric'].append(metric)
    results_dict_user['MAE'].append(avg_mae_user)
    results_dict_user['RMSE'].append(avg_rmse_user)

    # Item-based CF
    item_cf.sim_options['sim_options'] = {'name': metric}
    results_item = model_selection.cross_validate(item_cf, data, measures=['MAE', 'RMSE'], cv=5, verbose=False)
    avg_mae_item = np.mean(results_item['test_mae'])
    avg_rmse_item = np.mean(results_item['test_rmse'])
    results_dict_item['Metric'].append(metric)
    results_dict_item['MAE'].append(avg_mae_item)
    results_dict_item['RMSE'].append(avg_rmse_item)

# Plot the results side by side for both User-based and Item-based CF
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Bar graphs for User-based CF
ax[0, 0].bar(results_dict_user['Metric'], results_dict_user['MAE'], color='blue', alpha=0.7)
ax[0, 0].set_title('User-based CF - MAE')
ax[0, 0].set_xlabel('Similarity Metric')
ax[0, 0].set_ylabel('Performance')

ax[0, 1].bar(results_dict_user['Metric'], results_dict_user['RMSE'], color='orange', alpha=0.7)
ax[0, 1].set_title('User-based CF - RMSE')
ax[0, 1].set_xlabel('Similarity Metric')
ax[0, 1].set_ylabel('Performance')

# Bar graphs for Item-based CF
ax[1, 0].bar(results_dict_item['Metric'], results_dict_item['MAE'], color='green', alpha=0.7)
ax[1, 0].set_title('Item-based CF - MAE')
ax[1, 0].set_xlabel('Similarity Metric')
ax[1, 0].set_ylabel('Performance')

ax[1, 1].bar(results_dict_item['Metric'], results_dict_item['RMSE'], color='red', alpha=0.7)
ax[1, 1].set_title('Item-based CF - RMSE')
ax[1, 1].set_xlabel('Similarity Metric')
ax[1, 1].set_ylabel('Performance')

plt.suptitle('Impact of Similarity Metrics on Collaborative Filtering')
plt.show()
