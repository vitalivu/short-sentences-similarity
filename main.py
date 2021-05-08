import io
import pickle
import random
from time import perf_counter

import numpy as np
import torch
from pyhocon import ConfigFactory
from quart import Quart, request
from quart_cors import cors
from sentence_transformers import SentenceTransformer, util

config = ConfigFactory.parse_file('application.conf')
model_name = config.get_string('ss_search.model')

app = Quart(__name__)
app = cors(app, allow_origin="*")
model = SentenceTransformer(model_name)


def normalize_phases(sentence):
    ph = config.get('ss_search.norm_phases')
    for k, v in ph.items():
        sentence = sentence.replace(k, v)
    return sentence


from inflector import Inflector

stop_words = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
     'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
     "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
     'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
     'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
     'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
     'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
     'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
     'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
     's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
     'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
     "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
     'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
     "wouldn't"])


def clean_text(sent):
    # Removing non ASCII chars
    sent = str(sent).replace(r'[^\x00-\x7f]', r' ')
    sent_norm = sent.lower()
    # Remove any punctuation characters
    for c in [",", "!", ".", "?", "'", '"', ":", ";", "[", "]", "{", "}", "<", ">"]:
        sent_norm = sent_norm.replace(c, " ")

    # Remove stop words and Singularize all the words
    tokens = sent_norm.split()
    tokens = [Inflector().singularize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)


def suggest_question():
    return random.choice(list(id_2_question_map.values()))


@app.route('/api/suggest')
async def suggest():
    return {'question': suggest_question()}


@app.route('/api/search')
async def query():
    if 'random' in request.args:
        question = suggest_question()
    else:
        question = request.args.get('q')
    time_t1 = perf_counter()
    similars = search_for_similar(question)
    time_t2 = perf_counter()
    print("Response in", "{:.4f}".format(time_t2 - time_t1), "seconds")
    return {'question': question,
            'similars': similars,
            'query_time': "{:.4f}".format(time_t2 - time_t1)}


def search_for_similar(query):
    print("### Top K most similar queries of :", query)
    query_embedding = model.encode(clean_text(query), convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    result = []
    for score, idx in zip(top_results[0], top_results[1]):
        print("{:.4f} with: {}".format(score, questions[idx]))
        qids = cleanidx_2_rawquestionid_map.get(idx.item())
        for qid in qids:
            print(" -", id_2_question_map[qid])
        result.append({
            'id': qid,
            'clean_text': questions[idx],
            'questions': [id_2_question_map[qid] for qid in qids],
            'score': "{:.4f}".format(score),
            'author': 'Anonymous',
        })
    return result


# by default, Pickle does not support load model to cpu
class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


data_file = config.get_string('ss_search.data-file')
data_version = config.get_int('ss_search.data-version')

t1 = perf_counter()
if data_version == '1':
    with open(data_file, "rb") as fIn:
        stored_data = CpuUnpickler(fIn).load()
        questions = stored_data['questions']
        embeddings = stored_data['embeddings']
else:
    with open(data_file, "rb") as fIn:
        stored_data = CpuUnpickler(fIn).load()
        id_2_question_map = stored_data['id_2_question_map']
        clean_questions = stored_data['clean_questions']
        cleanidx_2_rawquestionid_map = stored_data['cleanidx_2_rawquestionid_map']
        embeddings = stored_data['embeddings']
        questions = np.array(clean_questions)
t2 = perf_counter()

print("Took {:.2f} seconds to import model".format(t2 - t1))
top_k = min(10, len(embeddings))

if __name__ == "__main__":
    search_for_similar('What is the approx annual cost of living while studying in UIC Chicago, for an Indian student?')
    app.run(debug=False)
