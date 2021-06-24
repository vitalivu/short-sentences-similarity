Short sentences similarity
===
## Idea behind the model
see `bert-sentence-similarity.ipynb`

## Training model
Using quora dataset of duplicated question pairs
see `aai_semantic_similarity_for_short_sentences.ipynb`

## Run server with docker

1. Download trained model from [my google.com/drive](#)
   - `embeddings_map500k.pkl` the model trained with cross/cosine similarity
    - `xgboost_pair.pkl`: the model trained with xgboost 
2. Run server with `docker-compose`

```yaml
version: '3'
services:
  oxford3k-flashcard:
    image: vuthailinh/short-sentences-similarity:latest
    volumes:
    - "./embeddings_map500k.pkl:/data/embeddings_map500k.pkl"
    - "./xgboost_pair.pkl:/data/xgboost_pair.pkl"
    ports:
    - "5000:5000"
```


3. Start UI application, see [vitalivu/semantic-similarity-search-ui](https://github.com/vitalivu/semantic-similarity-search-ui)

#### Startup success log
```
2021-05-24 18:49:52,506 [INFO] sentence_transformers.SentenceTransformer Load pretrained SentenceTransformer: paraphrase-distilroberta-base-v1
2021-05-24 18:49:52,507 [INFO] sentence_transformers.SentenceTransformer Did not find folder paraphrase-distilroberta-base-v1
2021-05-24 18:49:52,507 [INFO] sentence_transformers.SentenceTransformer Search model on server: http://sbert.net/models/paraphrase-distilroberta-base-v1.zip
2021-05-24 18:49:52,507 [INFO] sentence_transformers.SentenceTransformer Load SentenceTransformer from folder: ~/.cache/torch/sentence_transformers/sbert.net_models_paraphrase-distilroberta-base-v1
2021-05-24 18:49:53,217 [INFO] sentence_transformers.SentenceTransformer Use pytorch device: cpu
2021-05-24 18:50:12,319 [INFO] sentence_transformers.cross_encoder.CrossEncoder Use pytorch device: cpu
[18:50:17] WARNING: ../src/gbm/gbtree.cc:348: Loading from a raw memory buffer on CPU only machine.  Changing predictor to auto.
[18:50:17] WARNING: ../src/gbm/gbtree.cc:355: Loading from a raw memory buffer on CPU only machine.  Changing tree_method to hist.
[18:50:17] WARNING: ../src/learner.cc:223: No visible GPU is found, setting `gpu_id` to -1
2021-05-24 18:50:17,892 [INFO] lvnet.nlp.sssearch Preload took 19.8146 seconds
2021-05-24 18:50:17,893 [INFO] lvnet.nlp.sssearch Took 5.5641 seconds to import model
2021-05-24 18:50:17,893 [INFO] lvnet.nlp.sssearch SSSearch started in 25.3862
 * Serving Quart app 'main'
 * Environment: production
 * Please use an ASGI server (e.g. Hypercorn) directly in production
 * Debug mode: False
 * Running on http://127.0.0.1:5000 (CTRL + C to quit)
[2021-05-24 18:50:17,961] Running on http://127.0.0.1:5000 (CTRL + C to quit)
2021-05-24 18:50:17,961 [INFO] quart.serving Running on http://127.0.0.1:5000 (CTRL + C to quit)
```

## APIs
See `main.py`