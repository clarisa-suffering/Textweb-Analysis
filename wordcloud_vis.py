# wordcloud_vis.py
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Load Data ---
df = pd.read_excel("reviews_with_sentiment.xlsx")

# --- Split berdasarkan sentiment ---
positive_reviews = df[df['sentiment_label'] == "positive"]['cleaned_review'].tolist()
negative_reviews = df[df['sentiment_label'] == "negative"]['cleaned_review'].tolist()
neutral_reviews  = df[df['sentiment_label'] == "neutral"]['cleaned_review'].tolist()

# --- Gabungkan tiap kelompok jadi satu string ---
text_pos = " ".join(positive_reviews)
text_neg = " ".join(negative_reviews)
text_neu = " ".join(neutral_reviews)

# --- Generate WordCloud ---
wc_pos = WordCloud(width=600, height=400, background_color="white", colormap="Greens").generate(text_pos)
wc_neg = WordCloud(width=600, height=400, background_color="white", colormap="Reds").generate(text_neg)
wc_neu = WordCloud(width=600, height=400, background_color="white", colormap="Blues").generate(text_neu)

# --- Plot dalam 1 figure ---
plt.figure(figsize=(18, 10))

plt.subplot(1, 3, 1)
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews", fontsize=16)

plt.subplot(1, 3, 2)
plt.imshow(wc_neg, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews", fontsize=16)

plt.subplot(1, 3, 3)
plt.imshow(wc_neu, interpolation="bilinear")
plt.axis("off")
plt.title("Neutral Reviews", fontsize=16)

plt.tight
