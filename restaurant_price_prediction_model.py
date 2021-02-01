
# Ignore warnings :
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import gc
import re, string


from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score


from scipy.sparse import hstack, vstack
import xgboost as xgb
from xgboost.sklearn import XGBClassifier 
from collections import Counter


def nbModel(train_X, test_X, train_Y, test_Y):
    """ Training Naive Bayes Model"""
    class_names = ['RestaurantsPriceRange2_1', 
                   'RestaurantsPriceRange2_2',
                   'RestaurantsPriceRange2_3', 
                   'RestaurantsPriceRange2_4']
    scores = []
    p = MultinomialNB()
    preds = np.zeros((test_X.shape[0], len(class_names)))
    
    for i, class_name in enumerate(class_names):
        train_target = train_Y[class_name] 
        cv_score = np.mean(cross_val_score(estimator = p , X= train_X,
                                           y = train_target, cv = 5, scoring = 'f1'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        p.fit(train_X, train_target)
        preds[:,i] = p.predict_proba(test_X)[:,1]
    t = metrics.classification_report(np.argmax(test_Y[class_names].values, axis = 1),np.argmax(preds, axis = 1))
    print(t)


def XgbModel(train_X, test_X, train_Y, test_Y):
    """ XgBoost Model Training"""
    class_names = ['RestaurantsPriceRange2_1', 
                   'RestaurantsPriceRange2_2', 
                   'RestaurantsPriceRange2_3', 
                   'RestaurantsPriceRange2_4']
    cv_scores = []
    
    preds = np.zeros((test_X.shape[0], len(class_names)))
    
    for i, class_name in enumerate(class_names):
        if(class_names!="RestaurantsPriceRange2_2"):
            ct=Counter(train_Y[class_name])
            xgb_params = {'eta': 0.3, 
                          'max_depth': 5, 
                          'subsample': 0.8, 
                          'colsample_bytree': 0.8, 
                          'scale_pos_weight' : ct[0]/ct[1], # undersampling Restaurant price range class 2 as its the highest amongst all
                          'objective': 'binary:logistic', 
                          'eval_metric': 'auc', 
                          'seed': 1036
                         }
        else:
            xgb_params = {'eta': 0.3, 
                          'max_depth': 5, 
                          'subsample': 0.8, 
                          'colsample_bytree': 0.8, 
                          'objective': 'binary:logistic', 
                          'eval_metric': 'auc', 
                          'seed': 1036
                         }
            
        d_train = xgb.DMatrix(train_X, train_Y[class_name])
        d_test = xgb.DMatrix(test_X, test_Y[class_name])
            
        watchlist = [(d_test, 'test')]
        model = xgb.train(xgb_params, d_train, 200, watchlist, verbose_eval=False, early_stopping_rounds=30)
        print("class Name: {}".format(class_name))
        print(model.attributes()['best_iteration'])
        cv_scores.append(float(model.attributes()['best_score']))
            
        print('CV score for class {} is {}'.format(class_name, cv_scores[i]))
        preds[:,i] =model.predict(d_test)
    
    t = metrics.classification_report(np.argmax(test_Y[class_names].values, axis = 1),np.argmax(preds, axis = 1))
    print(t)


"""I will be creating the price prediction model
on the sample data created during EDA. The choice is  
due to hardware limitation on my local machine"""

print("Reading Reviews & Business combined Data - Sample")


df_review = pd.read_csv("./data/Review_Business_data_Sample.csv", encoding = 'utf-8')


df_review.drop(['Unnamed: 0'], axis = 1, inplace = True)


# text cleansing function
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


#done due to hardware limitations on my local machine
print("\n\n taking sub sample of 20% rows for model training \n\n")


# set frac = .2 to use the entire sample
sample_data = df_review[["text", "NoiseLevel", "CoatCheck", "Alcohol", 
                         "BusinessAcceptsCreditCards", "GoodForKids", 
                         "RestaurantsDelivery", "RestaurantsTakeOut", "OutdoorSeating", 
                         "WiFi", "Is_chain", "days_from_today", "vader_comp_score", 
                         "super_score", "RestaurantsPriceRange2"]].sample(frac = 0.2, random_state = 2308)

x_cols = ["text", "NoiseLevel", "CoatCheck", "Alcohol", "BusinessAcceptsCreditCards", 
          "GoodForKids", "RestaurantsDelivery", "RestaurantsTakeOut", "OutdoorSeating", 
          "WiFi", "Is_chain", "days_from_today", "vader_comp_score", "super_score"]

print("\n\n train test split \n\n")
x_train, x_test, y_train, y_test = train_test_split(sample_data[x_cols],
                                                    sample_data["RestaurantsPriceRange2"], 
                                                    test_size=0.2, random_state=2047)


cols_for_dummy = ["NoiseLevel", "CoatCheck", "Alcohol", "BusinessAcceptsCreditCards", 
                  "GoodForKids", "RestaurantsDelivery", "RestaurantsTakeOut", "OutdoorSeating", 
                  "WiFi", "Is_chain"]


print("\n\n Creating dummies for categorical variables for train & test data \n\n")


x_train = pd.get_dummies(x_train, columns = cols_for_dummy)


x_test = pd.get_dummies(x_test, columns = cols_for_dummy)


y_train = pd.DataFrame(y_train)
y_train  = pd.get_dummies(y_train, columns = ["RestaurantsPriceRange2"])


y_test = pd.DataFrame(y_test)
y_test  = pd.get_dummies(y_test, columns = ["RestaurantsPriceRange2"])


#x_train.columns


print("\n\n Building Tf-IDF vectors for text columns for both train & test data \n\n")


max_features = 2000 # selecting 2000 features due to hardware limitations on my local machine
tfidf = TfidfVectorizer(max_features = max_features,
                        analyzer='word',
                        stop_words= 'english', 
                        tokenizer=tokenize,
                        strip_accents='unicode',
                        ngram_range=(1,3))


get_ipython().run_cell_magic('time', '', "trn_term_doc = tfidf.fit_transform(x_train['text'])\ntest_term_doc = tfidf.transform(x_test['text'])")


print("\n\n Building sparse matrix of other featues in train & test data \n\n")


other_ftrs_forNB = ['days_from_today', 'super_score','NoiseLevel_None', 'NoiseLevel_average', 'NoiseLevel_loud',
                    'NoiseLevel_quiet', 'NoiseLevel_very_loud', 'CoatCheck_False','CoatCheck_True', 
                    'Alcohol_NoAlcohol', 'Alcohol_None','Alcohol_beer_and_wine', 'Alcohol_full_bar',
                    'BusinessAcceptsCreditCards_False', 'BusinessAcceptsCreditCards_True', 'GoodForKids_False', 
                    'GoodForKids_True', 'RestaurantsDelivery_False','RestaurantsDelivery_True', 
                    'RestaurantsTakeOut_False', 'RestaurantsTakeOut_True', 'OutdoorSeating_False',
                    'OutdoorSeating_True', 'WiFi_free', 'WiFi_no', 'WiFi_paid','Is_chain_False', 'Is_chain_True']


other_ftrs_forXGB = ['days_from_today', 'super_score', 'vader_comp_score','NoiseLevel_None', 'NoiseLevel_average', 
                     'NoiseLevel_loud','NoiseLevel_quiet', 'NoiseLevel_very_loud', 'CoatCheck_False','CoatCheck_True', 
                    'Alcohol_NoAlcohol', 'Alcohol_None','Alcohol_beer_and_wine', 'Alcohol_full_bar',
                    'BusinessAcceptsCreditCards_False', 'BusinessAcceptsCreditCards_True', 'GoodForKids_False', 
                    'GoodForKids_True', 'RestaurantsDelivery_False','RestaurantsDelivery_True', 
                    'RestaurantsTakeOut_False', 'RestaurantsTakeOut_True', 'OutdoorSeating_False',
                    'OutdoorSeating_True', 'WiFi_free', 'WiFi_no', 'WiFi_paid','Is_chain_False', 'Is_chain_True']


sp_oth_trn_ftrs_NB = sparse.csr_matrix(x_train[other_ftrs_forNB])
sp_oth_tst_ftrs_NB = sparse.csr_matrix(x_test[other_ftrs_forNB])


sp_oth_trn_ftrs_XGB = sparse.csr_matrix(x_train[other_ftrs_forXGB])
sp_oth_tst_ftrs_XGB = sparse.csr_matrix(x_test[other_ftrs_forXGB])


""" 
    We'll conduct 2 experiments
    one where we'll try to create price prediction model only with TF-IDF features
    and other where we'll mix TF-IDF with other features as well
    
    We'll try 2 models for both the experiments
    1) Naive Bayes model
    2) XgBoost Classifier model
    
    We'll use F1 score as model scoring metric
"""


print("\n\n Experiment 1 - NB model with only text features \n\n")


nbModel(trn_term_doc, test_term_doc, y_train, y_test)


print("\n\nExperiment 2 - NB model with  text features +  additional features \n\n")


nbModel( hstack([trn_term_doc,sp_oth_trn_ftrs_NB]), 
        hstack([test_term_doc,sp_oth_tst_ftrs_NB]), 
        y_train, y_test)


print("\n\nExperiment 3 - XGBoost model with  text features  \n\n")


XgbModel(trn_term_doc, test_term_doc, y_train, y_test)


print("\n\nExperiment 4 - XGBoost model with  text features + other features \n\n")


XgbModel( hstack([trn_term_doc,sp_oth_trn_ftrs_XGB]), 
         hstack([test_term_doc,sp_oth_tst_ftrs_XGB]), 
         y_train, y_test)




