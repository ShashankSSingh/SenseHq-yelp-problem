from sklearn.metrics import mean_squared_error
import math
import numpy as np

################################################
# constants for file handlers

#file_city_name = 'las_vegas'
#city_name = 'Las Vegas'

file_city_name = ['toronto','las_vegas', 'phoenix']
city_name = ['Toronto', 'Las Vegas', 'Phoenix']


# we only need the restaurants or users with more than this amount of reviews
NUM_REVIEWS_THRESH = 20


################################################
# utility functions

# computing the error
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return math.sqrt(mean_squared_error(prediction, ground_truth))

def list_rmse(prediction, ground_truth):
	return math.sqrt(mean_squared_error(prediction, ground_truth))

# def mape(prediction, ground_truth):
#     prediction = prediction[ground_truth.nonzero()].flatten() 
#     ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#     return np.mean(np.abs((ground_truth - prediction) / ground_truth)) * 100

