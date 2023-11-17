import matplotlib.pyplot as plt
import numpy as np
from surprise import Dataset, Reader, model_selection
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms.knns import KNNBasic

file_path = 'archive/ratings_small.csv'

reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(file_path, reader)

# Define the range of neighbors to be tested
neighbors_range = [5, 10, 15, 20, 25]

# Create a parameter grid for grid search
param_grid = {'k': neighbors_range}

# Use KNNBasic class directly for the algorithms
user_cf = KNNBasic(sim_options={'user_based': True})
item_cf = KNNBasic(sim_options={'user_based': False})

# Perform grid search for User-based CF
grid_search_user = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=5)
grid_search_user.fit(data)

# Perform grid search for Item-based CF
grid_search_item = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=5)
grid_search_item.fit(data)

# Print the best K for User-based CF
print(f"Best K for User-based CF: {grid_search_user.best_params['rmse']['k']}")

# Print the best K for Item-based CF
print(f"Best K for Item-based CF: {grid_search_item.best_params['rmse']['k']}")
