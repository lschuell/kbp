import re
import os
from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, Phraser
from collections import Counter
from googletrans import Translator
import faiss
from sklearn import preprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile
import logging


#------------------------------Evaluation------------------------------------

def evaluate(test_matrix_src, test_matrix_tgt, W, normalize=True, precisions=[1, 5, 10]):
    test_matrix_tgt_projected = test_matrix_src @ W
    test_matrix_tgt_base = test_matrix_tgt

    # L2-norm normalization and Euclidean distance search are quadratically proportional to cosine distance
    # -> norm=True (cosine distance/similarity), norm=False (euclidean distance)
    if normalize:
        test_matrix_tgt_projected = preprocessing.normalize(test_matrix_tgt_projected, axis=1, norm='l2')
        test_matrix_tgt_base = preprocessing.normalize(test_matrix_tgt_base, axis=1, norm='l2')

    size_tgt = test_matrix_tgt_projected.shape[1]
    index = faiss.IndexFlatL2(size_tgt)
    index.add(test_matrix_tgt_base.astype(np.float32))
    D, I = index.search(test_matrix_tgt_projected.astype(np.float32), 20)

    topks = np.zeros(len(precisions))
    for i, k in enumerate(precisions):
        for j in range(I.shape[0]):
            if j in I[j, :k]:
                topks[i] += 1

    topks /= (I.shape[0])

    for i, prec in enumerate(precisions):
        print("The precision@", prec, "is:", np.round(topks[i], 4))

    return topks

#----------------------------Lexicon---------------------------

def get_lexicon(word_vector_tgt: KeyedVectors, src="en", tgt="es", k=5000):
    '''
    Given source and target language, returns a lexicon for the k most frequent
    source words plus additional 1000 for testing using Google Translate
    '''
    # read source corpus
    with open("training-monolingual/news.2011.{}.shuffled.tokenized.lowercased.preprocessed.collocation".format(src),
              'r') as src_file:
        src_corpus = []
        for i, line in enumerate(src_file):
            src_corpus += line.split()

    # get top k frequent words and translations via GT
    counts = Counter(src_corpus)
    counts_sorted = [(l, k) for k, l in sorted([(j, i) for i, j in counts.items()], reverse=True)]
    sorted_words = [wc[0] for wc in counts_sorted]
    lexicon = []

    errors = 0
    success = 0

    for word in sorted_words:
        translator = Translator()
        translation = translator.translate(word, src=src, dest=tgt).text

        try:
            if isinstance(word_vector_tgt[translation.lower()], np.ndarray):
                lexicon.append((word, translation.lower()))
                success += 1
                if success % (k / 10) == 0:
                    print("Success:", success)

        except KeyError:
            errors += 1

        if success == k + 1000:
            break

    print("Total errors", errors)
    # get training lexicon for linear projection and test lexicon for evaluation
    train_lexicon = lexicon[:k]
    test_lexicon = lexicon[k:]

    return train_lexicon, test_lexicon

 #-------------------------Preprocessing-----------------------------

def preprocess(filepath):
    '''Creates corpus remedied for punctuation and numerical variation'''
    file = open(filepath, 'rt')
    corpus = file.read()
    file.close()

    # remove punctuation
    del_tokens = str.maketrans('', '', '!"#$%&\'()*+-/:;<=>?@[\\]^_`{|}~')
    corpus = corpus.translate(del_tokens)

    # unification of numerical characters
    corpus = re.sub(r"[-+]?\d*\.\d+|\d+", "NUM", corpus)

    outfile = open(filepath + ".preprocessed", "w")

    for line in corpus:
        outfile.write(line)

    outfile.close()


def collocation(filepath):
    '''Creates corpus considering collocations, frequent co-occuring bigrams are merged (new york -> new_york)'''

    abs_path = os.getcwd() + "/"
    corpus = Text8Corpus(datapath(abs_path + filepath))
    phrases = Phrases(corpus)
    collocations = Phraser(phrases)
    text_list = [collocations[line] for line in corpus]
    flattened_list = [i for sub in text_list for i in sub]
    flattened_corpus = " ".join(flattened_list)

    outfile = open(filepath + ".collocation", "w")

    for line in flattened_corpus:
        outfile.write(line)

    outfile.close()

#--------------------------------------------------------------------------

def get_emb_matrices(word_vector_src : KeyedVectors, word_vector_tgt : KeyedVectors, lexicon, size_src=800, size_tgt=200, norm_bf=True):
    k = len(lexicon)
    emb_matrix_src = np.zeros((k, size_src))
    emb_matrix_tgt = np.zeros((k, size_tgt))

    for i, src_tgt in enumerate(lexicon):
        emb_matrix_src[i, :] = word_vector_src[src_tgt[0]]
        emb_matrix_tgt[i, :] = word_vector_tgt[src_tgt[1]]

    if norm_bf:
        emb_matrix_src = preprocessing.normalize(emb_matrix_src, axis=1, norm='l2')
        emb_matrix_tgt = preprocessing.normalize(emb_matrix_tgt, axis=1, norm='l2')

    return emb_matrix_src, emb_matrix_tgt

def get_KeyVec(src = "en", tgt = "es", size_src=800, size_tgt=200, window=10, create=False, save=True, KG=False):
    '''
    :param src: source language
    :param tgt: target language
    :param size_src: embedding dimension for source language
    :param size_tgt: embedding dimension for target language
    :param window: window size
    :param create: bool - set to True if you want to create KeyedVectors, else KeyedVectors will be loaded
    :param save: bool - if create=True -> set save=True to save created KeyedVectors
    :return: source & target language KeyedVectors
    '''
    abs_path = os.getcwd() + "/"
    if KG:
        datapath_src = 'rwalks/rwalk_' + src + ".txt"
        datapath_tgt = 'rwalks/rwalk_' + tgt + ".txt"

    else:
        datapath_src = 'training-monolingual/news.2011.{}.shuffled.tokenized.lowercased.preprocessed.collocation'.format(src)  # , limit=100000)
        datapath_tgt = 'training-monolingual/news.2011.{}.shuffled.tokenized.lowercased.preprocessed.collocation'.format(tgt)  # , limit=100000)

    source_sentences = LineSentence(datapath_src)  # , limit=100000)
    target_sentences = LineSentence(datapath_tgt)  # , limit=100000)

    if not os.path.exists("training-monolingual/kv"):
        os.makedirs("training-monolingual/kv")

    kv_filename_src = abs_path + "training-monolingual/kv/keyvec_src-{}_dim{}_ws{}.kv".format(src, size_src, window)
    kv_filename_tgt = abs_path + "training-monolingual/kv/keyvec_tgt-{}_dim{}_ws{}.kv".format(tgt, size_tgt, window)
    vecfile_src = get_tmpfile(kv_filename_src)
    vecfile_tgt = get_tmpfile(kv_filename_tgt)

    if create:
        source_embedding = Word2Vec(source_sentences, size = size_src, window = window)  # generates the word2vec embeddings
        target_embedding = Word2Vec(target_sentences, size = size_tgt, window = window)  # generates the word2vec embeddings
        kv_src = source_embedding.wv
        kv_tgt = target_embedding.wv


        if save:
            kv_src.save(vecfile_src)
            kv_tgt.save(vecfile_tgt)

    else:
        assert(os.path.isfile(kv_filename_src) & os.path.isfile(kv_filename_tgt))
        kv_src = KeyedVectors.load(vecfile_src, mmap='r')
        kv_tgt = KeyedVectors.load(vecfile_tgt, mmap='r')

    return kv_src, kv_tgt