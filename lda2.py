import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import nltk
import pandas as pd

nltk.download('stopwords')
stop_words = stopwords.words('english')

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
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=42,
                                           passes=10)
    return lda_model, corpus, id2word


# --- Load your dataset ---
df = pd.read_excel("reviews_with_sentiment.xlsx")

# --- Split Data by Sentiment ---
positive_reviews = df[df['sentiment_label']=="Positive"]['cleaned_review'].tolist()
negative_reviews = df[df['sentiment_label']=="Negative"]['cleaned_review'].tolist()
neutral_reviews  = df[df['sentiment_label']=="Neutral"]['cleaned_review'].tolist()

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