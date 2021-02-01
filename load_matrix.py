import numpy as np
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import train_test_split
import pandas as pd
import utils
import json
from getarguments import file_city_name

#file_city_name = utils.file_city_name

header = ['user_id', 'business_id', 'weighted_rating', 'rating', 'sentiment_score']
df = pd.read_csv('./data/' + file_city_name + '_reviews_ratings_only.txt', sep='\t', names=header)

#print(df.head())
#print(df["weighted_rating"].describe())
#print(df["rating"].describe())

n_users = df.user_id.unique().shape[0]
n_items = df.business_id.unique().shape[0]

print('Number of users = ' + str(n_users) + ' | Number of items = ' + str(n_items))

print('load the index mappings')
with open('./data/' + file_city_name + '_filtered_user_index.json', 'r', encoding = 'utf-8') as user_idx_file:
    user_id_map = json.load(user_idx_file)
with open('./data/' + file_city_name + '_filtered_restaurants_index.json', 'r', encoding = 'utf-8') as rest_idx_file:
    item_id_map = json.load(rest_idx_file)

print('split and construct matrix')
train_data, test_data = train_test_split(df, test_size=0.25)

train_data_matrix = np.zeros((n_users, n_items))

total_count = 0
#print(train_data.head())

for line in train_data.itertuples():
    #print(total_count)
    #print(type(train_data_matrix[user_id_map[line[1]], item_id_map[line[2]]]))
    #train_data_matrix[user_id_map[line[1]], item_id_map[line[2]]] = line[3]  
    train_data_matrix[user_id_map[line[1]], item_id_map[line[2]]] = line[4]  
    total_count += 1

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    #test_data_matrix[user_id_map[line[1]], item_id_map[line[2]]] = line[3]
    test_data_matrix[user_id_map[line[1]], item_id_map[line[2]]] = line[4]
    total_count += 1

print('total users:', n_users)
print('total items:', n_items)
print('total reviews:', total_count)
print('sparsity:', total_count / (n_users * n_items * 1.0))


