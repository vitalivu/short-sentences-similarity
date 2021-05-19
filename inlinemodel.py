import math
from collections import Counter

import numpy  as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util

stops = set(stopwords.words("english"))
wnl = WordNetLemmatizer()

cross_encoder = CrossEncoder('cross-encoder/quora-distilroberta-base')
sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
stops = set(stopwords.words("english"))

df_train = pd.read_csv('../input/train.csv.zip')
df_train.head()
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

custom_stops = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                "you'd",
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
                'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
                'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                'ain',
                'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                "won't",
                'wouldn', "wouldn't"}


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


eps = 5000
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}


def extract_features(q1, q2):
    wms = word_match_share(q1, q2)
    tfi = tfidf_word_match_share(q1, q2)
    crs = cross_sim(q1, q2)
    cos = cosine_sim(q1, q2)
    return wms if not math.isnan(wms) else 0,\
           tfi if not math.isnan(tfi) else 0,\
           crs if not math.isnan(crs) else 0,\
           cos if not math.isnan(cos) else 0,


def word_match_share(q1, q2):
    q1words = {}
    q2words = {}
    for word in str(q1).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(q2).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    return (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))


def tfidf_word_match_share(q1, q2):
    q1words = {}
    q2words = {}
    for word in str(q1).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(q2).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    return np.sum(shared_weights) / np.sum(total_weights)


def cosine_sim(q1, q2):
    embeddings1 = sbert_model.encode([clean_text(q1)], convert_to_tensor=True)
    embeddings2 = sbert_model.encode([clean_text(q2)], convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores[0][0].item()


def cross_sim(q1, q2):
    cross_scores = cross_encoder.predict([[clean_text(q1), clean_text(q2)]])
    return cross_scores[0].item()


def clean_text(text):
    text = str(text).lower().replace(r'[^\x00-\x7f]', r' ')
    for c in [",", "!", ".", "?", '"', ":", ";", "[", "]", "{", "}", "<", ">"]:
        text = text.replace(c, " ")
    tokens = text.split(" ")
    tokens = [wnl.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stops and word not in custom_stops]
    return " ".join(tokens)
