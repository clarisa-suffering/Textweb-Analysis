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
    'like', 'liked', 'one', 'good', 'really', 'used', 'time', 'times', 'many', 'sunscreen','sunscreens', 
    'skin', 'best', 'favorite', 'great', 'love', 'well', 'day', 'sun', 'use', 'leave', 'leaves', 
    'tried', 'always', 'ever', 'go', 'much', 'product', 'using', 'cream', 'nice', 'buy', 'think', 'bad'
    'perfect', 'round', 'joseon', 'beauty', 'face', 'little', 'bit', 'without', 'apply', 'applied', 'applying',
     'non', 'lab', 'also', 'bought', 'dalba', 'hehe', 'lot', 'first', 'want', 'would', 'worst', 'satisfied', 'feel', 'even'
}
stop_words = stop_words.union(custom_stopwords)

# --- Files to analyze ---
files = {
    "Beauty of Joseon": "reviews_with_sentiment_boj.xlsx",
    "D'Alba": "reviews_with_sentiment_alba.xlsx"
}

results = {}

# LOAD ORIGINAL DATA
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


all_wordclouds = {}

for product_name, file in files.items():
    df = pd.read_excel(file)
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
    print(f"\n=== {product_name} ===")
    print("Positive Topics:")
    print(lda_pos.print_topics())

    print("\nNegative Topics:")
    print(lda_neg.print_topics())

    print("\nNeutral Topics:")
    print(lda_neu.print_topics())


    # --- Generate WORDCLOUDS ---
    wc_pos = WordCloud(width=600, height=400, background_color="white",
                       colormap="Greens", stopwords=stop_words).generate(text_pos)
    wc_neg = WordCloud(width=600, height=400, background_color="white",
                       colormap="Reds", stopwords=stop_words).generate(text_neg)
    wc_neu = WordCloud(width=600, height=400, background_color="white",
                       colormap="Blues", stopwords=stop_words).generate(text_neu)

    # Simpan ke dict
    all_wordclouds[product_name] = {
        "positive": wc_pos,
        "negative": wc_neg,
        "neutral": wc_neu
    }


# VISUALIZATION
# Sentiment Analysis (Pie Chart)
files = [
    "reviews_with_sentiment_boj.xlsx",
    "reviews_with_sentiment_alba.xlsx"
]

all_results = {}

for file in files:
    df = pd.read_excel(file)
    sentiment_percent = df['sentiment_label'].value_counts(normalize=True) * 100
    all_results[file] = sentiment_percent


# --- Satu Figure untuk Pie + Wordcloud ---
n_products = len(all_results)
ffig, axes = plt.subplots(1 + n_products, 3, figsize=(18, 6 * (1 + n_products)))

# Top section: 2 Pie charts for sentiment %
colors = {"positive":"green", "negative":"red", "neutral":"blue"}

custom_titles = [
    "Beauty of Joseon Sunscreen SA",
    "D'Alba Sunscreen SA"
]

for idx, (file, sentiment_percent) in enumerate(all_results.items()):
    product, wc_dict = list(all_wordclouds.items())[idx]
    axes[0, idx].pie(
        sentiment_percent,
        labels=sentiment_percent.index,
        autopct='%1.1f%%',
        colors=[colors.get(lbl, "grey") for lbl in sentiment_percent.index],
        startangle=90
    )
    axes[0, idx].set_title(custom_titles[idx])

axes[0, 2].axis("off")


# Bottom section: Aspect Analysis (Wordclouds)
for row, (product, wc_dict) in enumerate(all_wordclouds.items(), start=1):
    for col, sentiment in enumerate(["positive", "negative", "neutral"]):
        axes[row, col].imshow(wc_dict[sentiment], interpolation="bilinear")
        axes[row, col].axis("off")
        axes[row, col].set_title(f"{product} Sunscreen AA - {sentiment.capitalize()}")

plt.tight_layout()
plt.subplots_adjust(hspace=0.2)
plt.show()
plt.savefig("sentiment_wordclouds.png", bbox_inches="tight", dpi=150)
