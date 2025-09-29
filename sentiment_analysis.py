# 1. IMPORTS (all at the top)
import pandas as pd
import re
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt

# 2. DEFINE HELPER FUNCTIONS
def clean_review_text(text):
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)

    # Lowercase (optional: XLM-R is cased, but lowercase helps normalize)
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove emails (often junk in reviews)
    text = re.sub(r'\S+@\S+', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text



# --- MAIN SCRIPT ---

# 3. LOAD ORIGINAL DATA
print("Loading data...")
df = pd.read_excel('olive_young_global_reviews.xlsx')

# 4. CLEAN DATA (Step 2)
print("Cleaning text data...")
df['cleaned_review'] = df['review_text'].apply(clean_review_text)

# 5. LOAD SENTIMENT MODEL (Step 3)
print("Loading XLM-R sentiment model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# 6. PERFORM ANALYSIS
print("Running sentiment analysis...")
tqdm.pandas()
# Updated line 31
results = df['cleaned_review'].progress_apply(lambda x: sentiment_pipeline(str(x)[:512]))
# Add results to DataFrame
df['sentiment_label'] = [res[0]['label'] for res in results]
df['sentiment_score'] = [res[0]['score'] for res in results]

# 7. SAVE FINAL OUTPUT
print("Saving final results to Excel...")
df.to_excel('reviews_with_sentiment2.xlsx', index=False)

print("\nProcess complete!")

# Count the occurrences of each sentiment label
sentiment_counts = df['sentiment_label'].value_counts()

print("Overall Sentiment Distribution:")
print(sentiment_counts)

# Visualize the distribution with a bar chart
sentiment_counts.plot(kind='bar', color=['green', 'red', 'grey'])
plt.title('Sentiment Distribution of Product Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0) # Keep the labels horizontal
plt.show()

# Calculate the average rating for each sentiment label
avg_rating_by_sentiment = df.groupby('sentiment_label')['rating'].mean().sort_values(ascending=False)

print("\nAverage Star Rating by Sentiment:")
print(avg_rating_by_sentiment)