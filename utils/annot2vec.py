import pandas as pd
from gensim.models import word2vec
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import json

################################## Parameters ##################################
num_features = 50
min_word_count = 20
num_workers = 4
context = 10
downsampling = 1e-3
PATH_MODEL = 'annot_words.word2vec.model'
PATH_WEIGHT = 'word2weights.json.dict'
PATH_INPUT = '../dohki/data/gameknot/'
################################################################################

def annot_to_words(raw_annot, meaningless_words, stemmer):
    letters_only = re.sub('[^a-zA-Z]', ' ', str(raw_annot))
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in meaningless_words]
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    return stemming_words

def annot_to_words_fun():
    meaningless_words = set(stopwords.words('english'))
    stemmer = nltk.stem.PorterStemmer()
    return (lambda x: annot_to_words(x, meaningless_words, stemmer))

def w2v_plot(model):
    w2v = dict(zip(model.wv.index2word, model.wv.vectors))
    vocabs = list(w2v.keys())
    wv_list = [w2v[v] for v in vocabs]

    pca = PCA(n_components=2)
    xys = pca.fit_transform(wv_list)
    xs = xys[:, 0]
    ys = xys[:, 1]

    plt.figure(figsize=(8,6))
    plt.scatter(xs, ys, marker='o')
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy=(xs[i], ys[i]))
    plt.show()

if __name__ == "__main__":
    # cleaning annotations
    meaningless_words = set(stopwords.words('english'))
    stemmer = nltk.stem.PorterStemmer()

    print("[*] load files...")
    df = pd.DataFrame(columns=['move', 'fen', 'annotation'])
    for i in range(308):
        file_path = PATH_INPUT + 'gameknot_p%d.csv' % (i+1)
        try:
            _df_ = pd.read_csv(file_path, sep=',', usecols=['move','fen','annotation'], encoding='iso-8859-1')
            _df_ = _df_.dropna()
            df = df.append(_df_.iloc[:])
        except:
            continue
                                                    
    sentences = list()
    for raw_annot in df['annotation']:
        sentences.append(annot_to_words(raw_annot, meaningless_words, stemmer))

    print("[*] word embedding...")
    model = word2vec.Word2Vec(sentences,
                              workers=num_workers,
                              size=num_features,
                              sg=1,
                              min_count=min_word_count,
                              window=context,
                              sample=downsampling)
    model.init_sims(replace=True)
    model.save(PATH_MODEL)
    print("\t w2v model is saved at: " + PATH_MODEL)

    # tf-idf for weights
    print("[*] calculate tf-idf...")
    tfidfv = TfidfVectorizer()
    tfidfv.fit(list(map(' '.join, sentences)))
    max_idf = max(tfidfv.idf_)
    word2weight = defaultdict(
                lambda: max_idf,
                [(w, tfidfv.idf_[i]) for w, i in tfidfv.vocabulary_.items()])
    
    with open(PATH_WEIGHT, 'w') as f:
        json.dump(dict(word2weight), f)
    print("\t w2w weights is saved at: " + PATH_WEIGHT)

