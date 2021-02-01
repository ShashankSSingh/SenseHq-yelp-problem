import numpy as np
import pandas as  pd
import json
import utils
from getarguments import file_city_name
#file_city_name = utils.file_city_name


print('load the index mappings')
with open('./data/' + file_city_name + '_filtered_user_index.json', 'r', encoding = 'utf-8') as user_idx_file:
    user_id_map = json.load(user_idx_file)
with open('./data/' + file_city_name + '_filtered_restaurants_index.json', 'r', encoding = 'utf-8') as rest_idx_file:
    item_id_map = json.load(rest_idx_file)


user_id_map_df = pd.DataFrame.from_dict(user_id_map, orient='index').reset_index().rename(columns={'index' : 'user_id', 0 : 'user_idx'})
item_id_map_df = pd.DataFrame.from_dict(item_id_map, orient='index').reset_index().rename(columns={'index' : 'business_id', 0 : 'business_idx'})

print('load user based CF predictions')
cf_user_pred = np.load('./data/' + file_city_name + 'memory_based_cf_user' + '_all_predictions.np', allow_pickle = True)

print('load item based CF predictions')
cf_item_pred = np.load('./data/' + file_city_name + 'memory_based_cf_item' + '_all_predictions.np', allow_pickle = True)


header = ['user_id', 'business_id', 'weighted_rating', 'rating', 'sentiment_score']
df_ratings_only = pd.read_csv('./data/' + file_city_name + '_reviews_ratings_only.txt', sep='\t', names=header)


def flat_preds(preds):
    """ flatten the prediction matrix """
    preds = pd.DataFrame.from_records(preds)
    pred_flatten=pd.melt(preds.reset_index(),id_vars=['index']).sort_values(by=['index'])
    return pred_flatten
    


def map_pred_flatten(pred_flatten, user_id_map_df = user_id_map_df, item_id_map_df = item_id_map_df,  df = df_ratings_only):
 """ map the user and item indices 
    to actual user_ids & business_ids """   
 pred_flatten = pd.merge(pred_flatten, user_id_map_df, left_on = 'p_user_id', right_on = 'user_idx',how = 'left')
 pred_flatten = pd.merge(pred_flatten, item_id_map_df, left_on = 'p_business_id', right_on = 'business_idx',how = 'left')
 pred_flatten = pd.merge(pred_flatten, df[["user_id","business_id","rating"]], how='left', on = ["user_id", "business_id"])
 return pred_flatten


def get_top_n(predictions, n=10, minimumRating=4.0):
    """ get top n recommendations 
        by default gives top 10 recommendations
        recommendations are made only if the user has not 
        rated the business earlier & the predicted rating > 4
    """
    predictions = predictions[(predictions["predicted_rating"] > minimumRating) & (predictions["rating"].isnull()) ][["user_id", "business_id", "predicted_rating"]]
    
    table = pd.pivot_table(data=predictions,index=["user_id","business_id"])
    table = table.sort_values(['user_id',('predicted_rating')], ascending=[True,False]).reset_index()
    table = table.groupby('user_id').head(n).reset_index(drop=True)
    
    return table


""" user based recommendations """
print("Creating User based recommendations")
cf_user_pred_flatten = flat_preds(cf_user_pred).rename(columns={'index' : 'p_user_id', 'variable' : 'p_business_id', 'value' : 'predicted_rating'})


cf_user_pred_flatten = map_pred_flatten(cf_user_pred_flatten)


top_recommendations_user = get_top_n(cf_user_pred_flatten)


# top_recommendations_user[(top_recommendations_user["user_id"]=="TZQSUDDcA4ek5gBd6BzcjA") &
#                          (top_recommendations_user["business_id"]=="eremwEUrZUbg9FHMleD0_A") ]


""" item based recommendations """
print("Creating Item based recommendations")
cf_item_pred_flatten = flat_preds(cf_item_pred).rename(columns={'index' : 'p_user_id', 'variable' : 'p_business_id', 'value' : 'predicted_rating'})


cf_item_pred_flatten = map_pred_flatten(cf_item_pred_flatten)


top_recommendations_item = get_top_n(cf_item_pred_flatten)


# top_recommendations_item[(top_recommendations_item["user_id"]=="TZQSUDDcA4ek5gBd6BzcjA") &
#                          (top_recommendations_item["business_id"]=="eremwEUrZUbg9FHMleD0_A") ]


# top_recommendations_item.head()

top_recommendations_user.to_csv('./data/' + file_city_name + '_memory_based_cf_user' + '_recommendations.csv')
top_recommendations_item.to_csv('./data/' + file_city_name + '_memory_based_cf_item' + '_recommendations.csv')



