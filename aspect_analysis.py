import gensim
import pandas as pd
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
custom_stopwords = {
    'like', 'one', 'good', 'really', 'feel', 'used', 'times', 'many',
    'sunscreen', 'skin', 'best', 'favorite', 'great', 'love'
}
stop_words = stop_words.union(custom_stopwords)

# LOAD ORIGINAL DATA
print("Loading data...")
df = pd.read_excel('reviews_with_sentiment.xlsx')

# --- Helper Functions ---
def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def run_lda(texts, num_topics=5):
    # Tokenize
    data_words = list(sent_to_words(texts))
    # Remove stopwords
    data_words = remove_stopwords(data_words)
    # Build dictionary & corpus
    id2word = corpora.Dictionary(data_words)
    corpus = [id2word.doc2bow(text) for text in data_words]
    # LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=42,
                                           passes=10)
    return lda_model, corpus, id2word

# Hapus baris yang NaN di kolom cleaned_review, pastikan string
df = df.dropna(subset=["cleaned_review"])
df["cleaned_review"] = df["cleaned_review"].astype(str)

# --- Split Data by Sentiment ---
positive_reviews = df[df['sentiment_label']=="positive"]['cleaned_review'].tolist()
negative_reviews = df[df['sentiment_label']=="negative"]['cleaned_review'].tolist()
neutral_reviews  = df[df['sentiment_label']=="neutral"]['cleaned_review'].tolist()


# --- Gabungkan ke text string ---
text_pos = " ".join(positive_reviews)
text_neg = " ".join(negative_reviews)
text_neu = " ".join(neutral_reviews)

# --- Run LDA for each sentiment ---
lda_pos, corpus_pos, id2word_pos = run_lda(positive_reviews, num_topics=5)
lda_neg, corpus_neg, id2word_neg = run_lda(negative_reviews, num_topics=5)
lda_neu, corpus_neu, id2word_neu = run_lda(neutral_reviews, num_topics=5)

# --- Print topics ---
print("POSITIVE TOPICS:")
print(lda_pos.print_topics())

print("\nNEGATIVE TOPICS:")
print(lda_neg.print_topics())

print("\nNEUTRAL TOPICS:")
print(lda_neu.print_topics())



# --- WORDCLOUD VISUALIZATION ---
wc_pos = WordCloud(width=600, height=400, background_color="white", colormap="Greens", stopwords=stop_words).generate(text_pos)
wc_neg = WordCloud(width=600, height=400, background_color="white", colormap="Reds", stopwords=stop_words).generate(text_neg)
wc_neu = WordCloud(width=600, height=400, background_color="white", colormap="Blues", stopwords=stop_words).generate(text_neu)

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

plt.tight_layout()
plt.show()