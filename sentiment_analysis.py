# 1. IMPORTS
import pandas as pd
import re
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt

def clean_review_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # buat HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # buat URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # buat mentions and hashtags
    text = re.sub(r'\S+@\S+', '', text)  # buat emails
    text = re.sub(r'\s+', ' ', text).strip()  # whitespace
    return text


def process_reviews(input_file, output_file):
    print(f"\nProcessing {input_file}...")
    
    df = pd.read_excel(input_file)
    
    print("Cleaning text data")
    df['cleaned_review'] = df['review_text'].apply(clean_review_text)
    
    print("Running sentiment analysis...")
    tqdm.pandas()
    results = df['cleaned_review'].progress_apply(lambda x: sentiment_pipeline(str(x)[:512]))
    
    df['sentiment_label'] = [res[0]['label'] for res in results]
    df['sentiment_score'] = [res[0]['score'] for res in results]
    
    # save
    print(f"Saving results to {output_file}...")
    df.to_excel(output_file, index=False)
    
    # distribution
    sentiment_counts = df['sentiment_label'].value_counts()
    print("Overall Sentiment Distribution:")
    print(sentiment_counts)
    
    # visualisasi
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
    plt.title(f'Sentiment Distribution - {input_file}')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=0)
    plt.show()
    
    # avg rating
    if 'rating' in df.columns:
        avg_rating_by_sentiment = df.groupby('sentiment_label')['rating'].mean().sort_values(ascending=False)
        print("\nAverage Star Rating by Sentiment:")
        print(avg_rating_by_sentiment)

print("Loading XLM-R sentiment model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# run loop sa
files = [
    ("olive_young_boj_reviews.xlsx", "reviews_with_sentiment_boj.xlsx"),
    ("olive_young_alba_reviews.xlsx", "reviews_with_sentiment_alba.xlsx")
]

for input_file, output_file in files:
    process_reviews(input_file, output_file)

print("\nAll processing complete!")
