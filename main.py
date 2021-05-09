import io
import logging
import pickle
import random
from time import perf_counter

import numpy as np
import torch
from inflector import Inflector
from pyhocon import ConfigFactory
from quart import Quart, request
from quart_cors import cors
from sentence_transformers import SentenceTransformer, util, CrossEncoder

config = ConfigFactory.parse_file('application.conf')
rootLogger = logging.getLogger()
rootLogger.setLevel(config.get_string('logging.root.level', 'INFO'))
logger = logging.getLogger("lvnet.nlp.sssearch")
logger.setLevel(config.get_string('logging.level', 'DEBUG'))
formatter = logging.Formatter(config.get_string('logging.pattern', default='%(asctime)s [%(levelname)s] %(message)s'))
ch = logging.StreamHandler()
ch.setFormatter(formatter)
# logger.addHandler(ch)
rootLogger.addHandler(ch)
app = Quart(__name__)
app = cors(app, allow_origin="*")
t0 = perf_counter()
model = SentenceTransformer(config.get_string('ss_search.bi-encoder-model'))
cross_encoder = CrossEncoder(config.get_string('ss_search.cross-encoder-model'))

stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
              'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
              "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
              "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
              'wouldn', "wouldn't"}


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
    time_t1 = perf_counter()
    if 'random' in request.args and request.args.get('random') is not False:
        question = suggest_question()
    else:
        question = request.args.get('q')
    if 'alt' in request.args and request.args.get('alt') is not False:  # cross encoder
        similars = search_for_similar_cross(question)
    else:
        similars = search_for_similar(question)
    time_t2 = perf_counter()
    logger.info("Response in %.4f seconds", time_t2 - time_t1)
    return {'question': question,
            'similars': similars,
            'query_time': "{:.4f}".format(time_t2 - time_t1)}


def search_for_similar_cross(query):
    logger.info("### Cross-Encoder Top K most similar queries of : %s", query)
    query_embedding = model.encode(clean_text(query), convert_to_tensor=True)

    cross_cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    cross_top_results = torch.topk(cross_cos_scores, k=top_100)  # select top 100
    sentence_2_idx_map = {questions[idx]: idx[0].item() for idx in zip(cross_top_results[1])}
    top_sentences = list(sentence_2_idx_map.keys())
    sentence_combinations = [[clean_text(query), sentence] for sentence in top_sentences]
    cos_scores = cross_encoder.predict(sentence_combinations)
    top_results = reversed(np.argsort(cos_scores))
    result = []
    for _, idx in zip(range(10), top_results):
        logger.info("%.4f with %s", cos_scores[idx], top_sentences[idx])
        cidx = sentence_2_idx_map.get(top_sentences[idx])
        qids = cleanidx_2_rawquestionid_map.get(cidx)
        for qid in qids:
            logger.debug(" - %s", id_2_question_map[qid])
            result.append({
                'id': qid,
                'clean_text': questions[cidx],
                'questions': [id_2_question_map[qid] for qid in qids],
                'score': "{:.4f}".format(cos_scores[idx].item()),
                'author': 'Anonymous',
            })
    return result


def search_for_similar(query):
    logger.info("### Bi-Encoder Top K most similar queries of : %s", query)
    query_embedding = model.encode(clean_text(query), convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    result = []
    for score, idx in zip(top_results[0], top_results[1]):
        logger.info("%.4f with: %s", score, questions[idx])
        qids = cleanidx_2_rawquestionid_map.get(idx.item())
        for qid in qids:
            logger.debug(" - %s", id_2_question_map[qid])
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

top_k = min(10, len(embeddings))
top_100 = min(100, len(embeddings))

if __name__ == "__main__":
    logger.info("Preload took %.4f seconds", t1 - t0)
    logger.info("Took %.4f seconds to import model", t2 - t1)
    logger.info("SSSearch started in %.4f", perf_counter() - t0)
    app.run(debug=config.get_bool("ss_search.debug-api"))
