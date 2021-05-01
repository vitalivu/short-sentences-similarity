import io
import logging
import pickle
from time import perf_counter

import torch
from quart import Quart, request
from sentence_transformers import SentenceTransformer, util

app = Quart(__name__)
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

stopwords = set(
    ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
     "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can',
     'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't",
     'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't",
     'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'i', 'if',
     'in', 'into', 'is', 'isn', "isn't", "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't",
     'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of',
     'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same',
     'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than',
     'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this',
     'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were',
     'weren', "weren't", 'which', 'while', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd",
     "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])


def clean_text(sent):
    # Removing non ASCII chars
    sent = str(sent).replace(r'[^\x00-\x7f]', r' ')

    # Replace some common paraphrases
    sent_norm = sent.lower() \
        .replace("how do you", "how do i") \
        .replace("how do we", "how do i") \
        .replace("how can we", "how can i") \
        .replace("how can you", "how can i") \
        .replace("how can i", "how do i") \
        .replace("really true", "true") \
        .replace("what are the importance", "what is the importance") \
        .replace("what was", "what is") \
        .replace("so many", "many") \
        .replace("would it take", "will it take")

    # Remove any punctuation characters
    for c in [",", "!", ".", "?", "'", '"', ":", ";", "[", "]", "{", "}", "<", ">"]:
        sent_norm = sent_norm.replace(c, " ")

    # Remove stop words
    tokens = sent_norm.split()
    tokens = [token for token in tokens if token not in stopwords]
    return " ".join(tokens)


# by default, Pickle does not support load model to cpu
class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


# model = SentenceTransformer('paraphrase-distilroberta-base-v1')


# on the fly
@app.route('/search')
async def query():
    question = request.args.get('q')
    logging.info("searching query=", question)
    return {'question': question,
            'similars': [{
                'score': 0.6720,
                'question': 'what cost living (monthly yearly) graduate student studying mit'
            }]}


if __name__ == "__main__":

    t1 = perf_counter()
    # Load sentences & embeddings from disc
    with open('embeddings.pkl', "rb") as fIn:
        stored_data = CpuUnpickler(fIn).load()
        # question1 = stored_data['sentences']
        # embeddings1 = stored_data['embeddings']
        question2 = stored_data['sentences2']
        embeddings2 = stored_data['embeddings2']

    t2 = perf_counter()

    print("Took {:.2f} seconds to import model".format(t2 - t1))

    queries = [
        'What is the approx annual cost of living while studying in UIC Chicago, for an Indian student?']  # example from question1

    top_5 = min(5, len(embeddings2))

    time_t1 = perf_counter()
    for query in queries:
        query_embedding = model.encode(clean_text(query), convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings2)[0]
        top_results = torch.topk(cos_scores, k=top_5)
        print("### Query:", query)
        print("Top 5 most similar queries:")
        for score, idx in zip(top_results[0], top_results[1]):
            print("({:.4f})".format(score), question2[idx])

    time_t2 = perf_counter()
    print("Compute cosine-similarity in", "{:.4f}".format(time_t2 - time_t1), "seconds")

    app.run(debug=True)
