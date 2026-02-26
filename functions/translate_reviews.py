import pandas as pd
import numpy as np
from googletrans import Translator
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

def preprocess_reviews(df):
    """
    Preprocesses review data by selecting relevant columns, dropping missing values, and identifying low IDHM regions.

    Args:
        df (pd.DataFrame): Merged dataset containing review information.

    Returns:
        pd.DataFrame: Processed review data with IDHM categories and unique reviews.
    """
    review_data = df[["review_comment_message", "IDHM_low"]].dropna()
    review_data['review_comment_message'] = review_data['review_comment_message'].astype(str)
    review_data = review_data.drop_duplicates()
    return review_data

def translate_reviews_to_english(review_data):
    """
    Translates the review comments from Portuguese to English.

    Args:
        review_data (pd.DataFrame): DataFrame containing reviews to be translated.

    Returns:
        list: List of dictionaries with original and translated reviews.
    """
    translator = Translator()
    translated_reviews = []

    for _, row in review_data.iterrows():
        try:
            translated_text = translator.translate(row['review_comment_message'], src='pt', dest='en').text
            translated_reviews.append({
                'review_comment_message': row['review_comment_message'],
                'IDHM': row['IDHM'],
                'customer_state': row['customer_state'],
                'review_comment_message_english': translated_text
            })
        except Exception as e:
            print(f"Error translating row: {e}")
    
    return translated_reviews

def save_translated_reviews_to_csv(translated_reviews, file_path):
    """
    Saves the translated reviews to a CSV file.

    Args:
        translated_reviews (list): List of dictionaries containing the translated review data.
        file_path (str): Path to the CSV file to save the translated reviews.
    """
    translated_df = pd.DataFrame(translated_reviews)
    translated_df.to_csv(file_path, index=False, encoding='utf-8')

def read_translated_reviews_from_csv(file_path):
    """
    Reads the translated reviews from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the translated reviews.

    Returns:
        pd.DataFrame: DataFrame containing the translated review data.
    """
    return pd.read_csv(file_path)

def generate_wordclouds(data, is_low_idhm):
    """
    Generates a word cloud for reviews based on the IDHM category (low or high).

    Args:
        data (pd.DataFrame): DataFrame containing the reviews with IDHM categories.
        is_low_idhm (bool): A boolean flag to select either low IDHM (True) or high IDHM (False) reviews.
    
    Returns:
        WordCloud: A WordCloud object generated based on the reviews for the selected IDHM category.
    """
    reviews = " ".join(data.loc[data['IDHM_low'] == is_low_idhm, 'review_comment_message_english'].dropna())

    coolwarm = plt.cm.coolwarm
    colormap = LinearSegmentedColormap.from_list(
        "low_coolwarm", coolwarm(np.linspace(0, 0.5, 100))
    ) if is_low_idhm else LinearSegmentedColormap.from_list(
        "high_coolwarm", coolwarm(np.linspace(0.5, 1, 100))
    )

    wordcloud = WordCloud(
        background_color="white",
        colormap=colormap,
        stopwords=ENGLISH_STOP_WORDS,
        width=800,
        height=400
    ).generate(reviews)

    return wordcloud
