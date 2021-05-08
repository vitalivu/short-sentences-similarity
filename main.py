import io
import pickle
from time import perf_counter

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


def normalize_phases(sentence):
    ph = config.get('ss_search.norm_phases')
    for k, v in ph.items():
        sentence = sentence.replace(k, v)
    return sentence


def clean_text(sent):
    # Removing non ASCII chars
    sent = str(sent).replace(r'[^\x00-\x7f]', r' ')

    # Replace some common paraphrases
    sent_norm = normalize_phases(sent.lower())

    # Remove any punctuation characters
    for c in [",", "!", ".", "?", "'", '"', ":", ";", "[", "]", "{", "}", "<", ">"]:
        sent_norm = sent_norm.replace(c, " ")

    # Remove stop words
    tokens = sent_norm.split()
    tokens = [token for token in tokens if token not in stopwords]
    return " ".join(tokens)


# on the fly
@app.route('/api/search')
async def query():
    question = request.args.get('q')
    time_t1 = perf_counter()
    similars = search_for_similar(question)
    time_t2 = perf_counter()
    print("Response in", "{:.4f}".format(time_t2 - time_t1), "seconds")
    return {'question': question,
            'similars': similars,
            'query_time': "{:.4f}".format(time_t2 - time_t1)}


def search_for_similar(query):
    print("### Query:", query)
    query_embedding = model.encode(clean_text(query), convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    result = []
    for score, idx in zip(top_results[0], top_results[1]):
        print("({:.4f})".format(score), questions[idx])
        result.append({
            'id': idx.item(),
            'title': questions[idx],
            'score': "{:.4f}".format(score),
            'author': 'Anonymous',
            # 'desc': "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin a lectus blandit, aliquam magna at, rhoncus nisi. Etiam ultrices, nunc in tempus volutpat, mi nisi tincidunt risus, vel consectetur nisl lorem vel risus. Aenean turpis ligula, consectetur id bibendum ac, maximus et libero. In porta, dui non tristique posuere, dui libero viverra enim, et rutrum odio nunc lobortis libero. In imperdiet purus eu vestibulum vestibulum. Nullam nec rutrum nisl. Sed euismod est sed congue tincidunt. Proin ornare elit aliquet nulla malesuada aliquam."
        })
    return result


# by default, Pickle does not support load model to cpu
class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


datafile = config.get_string('ss_search.datafile')
t1 = perf_counter()
with open(datafile, "rb") as fIn:
    stored_data = CpuUnpickler(fIn).load()
    questions = stored_data['questions']
    embeddings = stored_data['embeddings']
t2 = perf_counter()

print("Took {:.2f} seconds to import model".format(t2 - t1))
top_k = min(10, len(embeddings))

if __name__ == "__main__":
    search_for_similar('What is the approx annual cost of living while studying in UIC Chicago, for an Indian student?')
    app.run(debug=False)
