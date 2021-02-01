# Sensehq problem yelp price prediction and recommendation

## Exploratory Data Analysis:

### Code usage
Load `yelp_EDA.ipynb` into a jupyter notebook or a pyhton IDE of your choice
Execute all the cells

### What this code is doing?
Here the Business & Reviews data has been explored and features created for 
Restaurant's price modelling

The notebook explores & visualizes the relationship of various restaurant attributes,
both from business & reviews text with restaurant pricing.

A business establishment is considered restaurant only if the categories belong to 
one of the following - "Restaurants", "Food", "Bar". There are multiple cuisine type
tags in the categories but looking at them the rows which had cuisine type tags also 
had one of these tags. 

With this filtering there are approx 85K restaurants & after filtering for presence of 
"RestaurantPrice2" tag in attributes there are approx 72K restaurants.

Review data was fileterd for only these 72K restaurants which was approx 5.5 million rows
against 8.3 million rows of total reviews data set.

To create sentiment score features a random sample of 10% of rows (550K) was taken

Created sentiment score  features using VADER sentiment scoring. VADER is chosen as it 
is trained and tuned to social media reviews and is better at handling emoticons.
Also have used textblob sentiment scores. 

It was also seen that some of the ratings conveyed a different message when compared
to review text. This could be because of inherent bias of user's previous experiences
and hence a super score of stars was calculated as -
super_score = stars + (vader_compound_score * textblob_polarity_score)
It range is from 0 to 6 against star's range from 1 to 5

The output of this notebook is "Review_Business_data_Sample.csv" file which is a 
sampled version of business & reviews data. The choice is due to hardware
limitations on my local machine. This is a sample of 
This file has been used for Restaurant price modelling

From this EDA we discover that "Las Vegas", "Toronto" & "Phoenix" are top 3 cities 
by review counts and hence for the recommendation problem I have focussed on
these cities only. This has again been done owing to hardware limitations in 
my local machine


## Restaurant Price prediction model:

### Code usage
1. run `restaurant_price_prediction_model.py`

### What this code is doing?
It takes Review_Business_data_Sample.csv as the input. 
Here a ~20% sample of 550K rows for creating a TF-IDF features 
of the raw text data.

On looking at the restaurant price ratings the class - 
RestaurantPrice '4' (Premium restaurants) and 'RestaurantPrice '3' (High End restaurants)
are way less compared to '1' and '2' 
So in a way this is an imbalance classification problem

I conducted 2 set of experiments with 2 set of training algos-
    1) where I'll try to create price prediction model only with TF-IDF features
    2) other where we'll mix TF-IDF with other features as well
    
I tried 2 models for both the experiments-
    1) Naive Bayes model as baseline model
    2) XgBoost Classifier model 
    
I used F1 score as model scoring metric due to imbalance 

Out of all the experiments Xgboost classifier with TF-IDF + business features
gives the best F1 scores across all the classes 


## Collaborative Filtering:

## Code Usage

1. `utils.py` defines the list of  cities  we're currently dealing with. 
..*  Currently limited to "Las Vegas", "Toronto" & "Phoenix" due to hardware limitations on my local machine.
..* `file_city_name` defines the city name used to all of the intermediate files
..* `city_name` has to match the city name stored within the Yelp dataset

2. `getarguments.py` invoked in all the scripts & gets  the city name as user input from CLI
    if the city doesn't exists in the list defined in utils.py execution terminates

3. `extract_city_ratings.py` 
..* Find all the reviews from the city
..* Filter the users and restaurants with too few reviews (current threshold at 20)
..* Construct the skinny file which only include the `user_id`, `business_id` and the rating, 

4. `extract_matrix_index.py` defines the mappings from the `user_id` and `business_id` in the dataset, into integers from 0, used as the index for the data matrix

5. `load_matrix.py` should be imported for all of the learning algorithms, to load the training and testing data matrix, with the consistent index

6. `concat_reviews.py` concatenate all the reviews for an user or a restaurant into a long sentence

7. `dump_bag_of_words.py` use the concatenated reviews and construct the tf-idf vectors for each user and restaurant, should be called once before running any text-based or rating-based algorithms

8. `dump_user_preference.py` constructs the user preference matrix by computing the linear combination of restaurant feature vectors for each user, using the user's ratings as the weights, should be called at least once before running the rating-based algorithm

9. `text_based_recommendations.py` predicts the potential ratings of the restaurants by a user  purely on cosine similarity  of all the reviews  of the user & and all the reviews received by restaurant.

10. `memory_based_cf.py` predicts the potential ratings of the restaurants by user based on the user based and item based collaborative 
filtering algorithm

11. `pred_cf_recommend.py` recommends top n (default = 10) restaurants for each user based on the CF user based & item based predictions 

12. To execute the code -
For Data Prep-
 `extract_city_ratings.py` "City Name" -> `extract_matrix_index.py` "City Name" -> `concat_reviews.py` "City Name" -> `dump_bag_of_words.py` "City Name" -> `dump_user_preference.py` "City Name"

 For Model Training -
 `Data Prep` --> `text_based_recommendation.py` "City Name"
 `Data Prep` --> --> `memory_based_cf.py` "City Name"

For Model Scoring & getting top 20 recommendations per user (done only for CF algo as it was the best performing one)
`pred_cf_recommend.py` "City Name"