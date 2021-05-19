import io
import logging
import os
import pickle
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pyhocon import ConfigFactory
from quart import Quart, request, abort
from quart_cors import cors
from sentence_transformers import SentenceTransformer, util, CrossEncoder

from inlinemodel import clean_text, extract_features

if os.path.isfile('application.local.conf'):
    config = ConfigFactory.parse_file('application.local.conf').with_fallback('application.conf')
else:
    config = ConfigFactory.parse_file('application.conf')

rootLogger = logging.getLogger()
rootLogger.setLevel(config.get_string('logging.levels.root', 'INFO'))
logger = logging.getLogger("lvnet.nlp.sssearch")
logger.setLevel(config.get_string('logging.levels.lvnet.nlp.sssearch', 'DEBUG'))
formatter = logging.Formatter(config.get_string('logging.pattern', default='%(asctime)s [%(levelname)s] %(message)s'))
if config.get_bool('logging.appenders.console.enabled', True):
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    rootLogger.addHandler(ch)
if config.get_bool('logging.appenders.file.enabled', True):
    fh = logging.FileHandler(config.get_string("logging.appenders.file.file-name"))
    fh.setFormatter(formatter)
    rootLogger.addHandler(fh)

app = Quart(__name__)
app = cors(app, allow_origin="*")

t0 = perf_counter()
model = SentenceTransformer(config.get_string('ss_search.bi-encoder-model'))
cross_encoder = CrossEncoder(config.get_string('ss_search.cross-encoder-model'))


def suggest_question():
    return random.choice(all_questions)


@app.route('/api/suggest')
async def suggest():
    return {'question': suggest_question()}


@app.route('/api/compare')
async def compare():
    if 'q1' not in request.args or 'q2' not in request.args:
        return abort(400, description='Missing required parameters')

    qt1 = perf_counter()
    q1 = request.args.get('q1')
    q2 = request.args.get('q2')

    logger.info("Comparing %s to %s", q1, q2)
    x_test = pd.DataFrame()
    wms, tfi, crs, cos = extract_features(clean_text(q1), clean_text(q2))
    x_test['word_match'] = [wms]
    x_test['tfidf_word_match'] = [tfi]
    x_test['cross_sim'] = [crs]
    x_test['cosine_sim'] = [cos]
    x_test.fillna(0)
    d_test = xgb.DMatrix(x_test)

    y_est = bst.predict(d_test)
    qt2 = perf_counter()
    logger.info("Response in %.4f seconds", qt2 - qt1)
    return {'question1': q1,
            'question2': q2,
            'scores': [{'label': 'combination', 'score': y_est.item()},
                       {'label': 'word_match', 'score': wms},
                       {'label': 'tfidf_wms', 'score': tfi},
                       {'label': 'cross_sim', 'score': crs},
                       {'label': 'cosine_sim', 'score': cos}],
            'compare_time': "{:.4f}".format(qt2 - qt1)}


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
xgmodel_file = config.get_string('ss_search.xg-model-file')
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

with open(xgmodel_file, "rb") as fIn:
    stored_data = CpuUnpickler(fIn).load()
    bst = stored_data['bst']
    params = stored_data['params']

t2 = perf_counter()

top_k = min(10, len(embeddings))
top_100 = min(100, len(embeddings))
all_questions = list(id_2_question_map.values())
if __name__ == "__main__":
    logger.info("Preload took %.4f seconds", t1 - t0)
    logger.info("Took %.4f seconds to import model", t2 - t1)
    logger.info("SSSearch started in %.4f", perf_counter() - t0)
    app.run(debug=config.get_bool("ss_search.debug-api"))
