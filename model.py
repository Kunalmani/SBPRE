import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import  train_test_split
import pickle

import nltk

nltk.download('omw-1.4')

#Below pickel files were generated during model building in Google Colab drive.
#Copied the files to my local Spider project folder.
#All pickel files placed in foler called "models" , Data file placed in folder called "data" and html file placed in folder
#named "templates"

sentiment_model_file = './models/randome_forest_model.pkl'
recomm_model_file = './models/user_based_cf_recomm.pkl'
tfidf_vect_file = './models/tf_idf_vect.pkl'
product_sentiment = './models/product_sentiments.pkl'

class RecommendClass():

    # get sentiment score of the product
    def get_sentiment_score(self, recomm_df, product_name):
        prod_senti_model = pickle.load(open(product_sentiment, 'rb'))
        sentiment_score = prod_senti_model.loc[product_name].values[0]
        return sentiment_score

    # get 5 best product recommendations based on the sentiments
    def get_top_5_products(self, recomm_df, recommendations):
        recommendations['sentiment_score'] = [self.get_sentiment_score(recomm_df, x) for x in recommendations['product_name']]
        recommendations.sort_values(by='sentiment_score', ascending=False, inplace=True)
        top_recomm_products = recommendations['product_name'][0:5]
        return top_recomm_products

    # get product recommendations. This method is called from app.py 
    def recommend_products(self, name):
        recommendation_df = pd.read_csv("./data/sample30.csv")
        recommendation_df['user_id'] = pd.factorize(recommendation_df['reviews_username'])[0]
        recommendation_df.rename(columns={'id':'product_id', 'reviews_rating':'rating', 'reviews_username':'user_name'}, inplace=True)
        
        # get 20 recommendations the user without sentiment analysis. Out of this list , we will filter top five next
        usrid = recommendation_df[recommendation_df['user_name']==name]['user_id'].unique()
        recommendations = pickle.load(open(recomm_model_file, 'rb'))
        d = recommendations.loc[int(usrid)].sort_values(ascending=False)[0:20]
        recommended_product_20 = pd.DataFrame(d)
        recommended_product_20.reset_index(inplace=True)
        recommended_product_20['product_name'] = [recommendation_df[recommendation_df['product_id']==x]['name'].unique()[0] for x in recommended_product_20['product_id']]
        
        # get top 5 products for user
        top_5_products = self.get_top_5_products(recommendation_df, recommended_product_20)
        return list(top_5_products)
    

    def __init__(self) -> None:
        pass

    
