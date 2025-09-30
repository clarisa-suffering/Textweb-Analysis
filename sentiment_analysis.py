# 1. IMPORTS
import pandas as pd
import re
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt

# 2. HELPER FUNCTION TO CLEAN REVIEWS
def clean_review_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# 3. FUNCTION TO PROCESS A SINGLE FILE
def process_reviews(input_file, output_file):
    print(f"\nProcessing {input_file}...")
    
    # Load data
    df = pd.read_excel(input_file)
    
    # Clean reviews
    print("Cleaning text data...")
    df['cleaned_review'] = df['review_text'].apply(clean_review_text)
    
    # Sentiment analysis
    print("Running sentiment analysis...")
    tqdm.pandas()
    results = df['cleaned_review'].progress_apply(lambda x: sentiment_pipeline(str(x)[:512]))
    
    df['sentiment_label'] = [res[0]['label'] for res in results]
    df['sentiment_score'] = [res[0]['score'] for res in results]
    
    # Save results
    print(f"Saving results to {output_file}...")
    df.to_excel(output_file, index=False)
    
    # Distribution
    sentiment_counts = df['sentiment_label'].value_counts()
    print("Overall Sentiment Distribution:")
    print(sentiment_counts)
    
    # Visualization
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'grey'])
    plt.title(f'Sentiment Distribution - {input_file}')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=0)
    plt.show()
    
    # Average rating
    if 'rating' in df.columns:
        avg_rating_by_sentiment = df.groupby('sentiment_label')['rating'].mean().sort_values(ascending=False)
        print("\nAverage Star Rating by Sentiment:")
        print(avg_rating_by_sentiment)

# 4. LOAD SENTIMENT MODEL ONCE
print("Loading XLM-R sentiment model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# 5. RUN FOR MULTIPLE FILES
files = [
    ("olive_young_boj_reviews.xlsx", "reviews_with_sentiment_boj.xlsx"),
    ("olive_young_alba_reviews.xlsx", "reviews_with_sentiment_alba.xlsx")
]

for input_file, output_file in files:
    process_reviews(input_file, output_file)

print("\nAll processing complete!")
