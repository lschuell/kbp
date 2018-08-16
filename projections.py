import torch
import torch.nn as nn
from scipy.linalg import svd
from utils import *
import argparse
import pickle


class LinearProjection():

    def __init__(self, word_vector_src: KeyedVectors, word_vector_tgt: KeyedVectors, train_lexicon, test_lexicon,
                 size_src=800, size_tgt=200, norm_bf=False):
        '''
        Computes the linear projection between two word embeddings using Moore-Penrose-Pseudoinverse to solve least
        squares problem algebraically.
        See http://ruder.io/cross-lingual-embeddings/index.html#linearprojection
        https://arxiv.org/abs/1309.4168
        :param word_vector_source: word vector (keyedvector) of source language
        :param word_vector_target: word vector (keyedvector) of target language
        :param train_lexicon: iterable of (source_word, target_word) or (source_word, target_word, confidence) for training
        :param test_lexicon: iterable of (source_word, target_word) or (source_word, target_word, confidence) for testing
        '''

        self.word_vector_src = word_vector_src
        self.word_vector_tgt = word_vector_tgt
        self.train_lexicon = train_lexicon
        self.test_lexicon = test_lexicon

        self.train_matrix_src, self.train_matrix_tgt = get_emb_matrices(self.word_vector_src, self.word_vector_tgt,
                                                                        self.train_lexicon, size_src=size_src,
                                                                        size_tgt=size_tgt, norm_bf=norm_bf)
        self.test_matrix_src, self.test_matrix_tgt = get_emb_matrices(self.word_vector_src, self.word_vector_tgt,
                                                                      self.test_lexicon, size_src=size_src,
                                                                      size_tgt=size_tgt, norm_bf=norm_bf)
        logging.info("Embedding Matrices created")

        self.W = self.project()
        logging.info("Projection Matrix W created")

    def project(self):
        '''Project X to Z with matrix W using Moosre Penrose Pseudoinverse'''
        X_mpi = np.linalg.pinv(self.train_matrix_src)  # Moore Penrose Pseudoinverse
        W = np.dot(X_mpi, self.train_matrix_tgt)  # linear map matrix W
        return W


class SGD_Projection(LinearProjection):

    def __init__(self, word_vector_src: KeyedVectors, word_vector_tgt: KeyedVectors, train_lexicon, test_lexicon,
                 size_src=800, size_tgt=200, norm_bf=False):
        self.size_src = size_src
        self.size_tgt = size_tgt
        super(SGD_Projection, self).__init__(word_vector_src, word_vector_tgt, train_lexicon, test_lexicon, size_src,
                                             size_tgt, norm_bf)
        '''
        Computes the linear projection between two word embeddings using linear projection optimized by SGD.
        See http://ruder.io/cross-lingual-embeddings/index.html#linearprojection
        https://arxiv.org/abs/1309.4168
        :param word_vector_source: word vector (keyedvector) of source language
        :param word_vector_target: word vector (keyedvector) of target language
        :param train_lexicon: iterable of (source_word, target_word) or (source_word, target_word, confidence) for training
        :param test_lexicon: iterable of (source_word, target_word) or (source_word, target_word, confidence) for testing
        '''

    def project(self, epochs=3000, l_rate=0.01):
        '''Project X to Z with matrix W using SGD. Linear layer from Pytorch used
        with bias set to False so that only matrix weights learned, SGD as optimizer'''

        class LR_Model(nn.Module):

            def __init__(self, in_dim, out_dim):
                super(LR_Model, self).__init__()
                self.linear = nn.Linear(in_dim, out_dim, bias=False)

            def forward(self, x):
                out = self.linear(x)
                return out

        model = LR_Model(self.size_src, self.size_tgt)
        criterion = nn.MSELoss()  # MSE loss corresponds to objective function
        optimiser = torch.optim.SGD(model.parameters(), lr=l_rate)

        for epoch in range(epochs):
            epoch += 1
            optimiser.zero_grad()
            output = model.forward(torch.Tensor(self.train_matrix_src))
            loss = criterion(output, torch.Tensor(self.train_matrix_tgt))
            loss.backward()
            optimiser.step()
            if epoch % 100 == 0:
                print('epoch {}, loss {}'.format(epoch, loss.data[0]))

        W = list(model.parameters())[0]
        W = np.transpose(torch.tensor(W).detach().numpy())
        return W


class ORTH_Projection(LinearProjection):

    def __init__(self, word_vector_src: KeyedVectors, word_vector_tgt: KeyedVectors, train_lexicon, test_lexicon,
                 size_src=800, size_tgt=200, norm_bf=False):
        super(ORTH_Projection, self).__init__(word_vector_src, word_vector_tgt, train_lexicon, test_lexicon, size_src,
                                              size_tgt, norm_bf)
        '''
        Computes the orthogonal projection between two word embeddings.
        See http://ruder.io/cross-lingual-embeddings/index.html#linearprojection
        https://arxiv.org/abs/1309.4168
        :param word_vector_source: word vector (keyedvector) of source language
        :param word_vector_target: word vector (keyedvector) of target language
        :param train_lexicon: iterable of (source_word, target_word) or (source_word, target_word, confidence) for training
        :param test_lexicon: iterable of (source_word, target_word) or (source_word, target_word, confidence) for testing
        '''

    def project(self, delta=0.01):
        '''Project X to Z with matrix W using orthogonal constraint and gradient descent'''
        X = self.train_matrix_src.copy();
        Z = self.train_matrix_tgt.copy()

        if X.shape[1] < Z.shape[1]:
            X = np.concatenate((X, np.full((X.shape[0], Z.shape[1] - X.shape[1]), delta)), axis=1)
            X = preprocessing.normalize(X, axis=1, norm='l2')
            self.train_matrix_src = X

            X_test = self.test_matrix_src.copy()
            X_test = np.concatenate((X_test, np.full((X_test.shape[0], Z.shape[1] - X_test.shape[1]), delta)), axis=1)
            X_test = preprocessing.normalize(X_test, axis=1, norm='l2')
            self.test_matrix_src = X_test

        elif X.shape[1] > Z.shape[1]:
            Z = np.concatenate((Z, np.full((Z.shape[0], X.shape[1] - Z.shape[1]), delta)), axis=1)
            Z = preprocessing.normalize(Z, axis=1, norm='l2')
            self.train_matrix_tgt = Z

            Z_test = self.test_matrix_tgt.copy()
            Z_test = np.concatenate((Z_test, np.full((Z_test.shape[0], X.shape[1] - Z_test.shape[1]), delta)), axis=1)
            Z_test = preprocessing.normalize(Z_test, axis=1, norm='l2')
            self.test_matrix_tgt = Z_test

        M = np.dot(np.transpose(X), Z)
        U, S, V_T = svd(M, full_matrices=True)
        S_1s = np.diag(np.append(np.ones((S > 0).sum()), (np.zeros(min(U.shape[1], V_T.shape[0]) - (S > 0).sum()))))
        W = U @ S_1s @ V_T

        return W


def main():
    # create fully preprocessed files
    if args.create_pre:
        preprocess('training-monolingual/news.2011.{}.shuffled.tokenized.lowercased'.format(args.src))
        preprocess('training-monolingual/news.2011.{}.shuffled.tokenized.lowercased'.format(args.tgt))
    if args.create_col:
        collocation("training-monolingual/news.2011.{}.shuffled.tokenized.lowercased.preprocessed".format(args.src))
        collocation("training-monolingual/news.2011.{}.shuffled.tokenized.lowercased.preprocessed".format(args.tgt))

    source_kv, target_kv = get_KeyVec(src=args.src, tgt=args.tgt, size_src=args.size_src, size_tgt=args.size_tgt,
                                      create=args.create_kv, save=args.save_kv, KG=args.kg)

    if args.create_lexicon:
        train_lexicon, test_lexicon = get_lexicon(target_kv, args.src, args.tgt, k=768)
        with open("./training-monolingual/lexicon/train_lexicon_" + args.src + "-" + args.tgt, "wb") as train:
            pickle.dump(train_lexicon, train, protocol=pickle.HIGHEST_PROTOCOL)
        with open("./training-monolingual/lexicon/test_lexicon_" + args.src + "-" + args.tgt, "wb") as test:
            pickle.dump(test_lexicon, test, protocol=pickle.HIGHEST_PROTOCOL)


    if args.get_result:
        with open("./training-monolingual/lexicon/train_lexicon_" + args.src + "-" + args.tgt, "rb") as train:
            train_lexicon = pickle.load(train)
        with open("./training-monolingual/lexicon/test_lexicon_" + args.src + "-" + args.tgt, "rb") as test:
            test_lexicon = pickle.load(test)

        if args.sub_lexicon:
            train_lexicon, test_lexicon = get_sublexicon(train_lexicon, test_lexicon, k=768, test_size=500)

        logging.info("Linear projection with Moore-Penrose-Pseudoinverse")
        lp_MPP = LinearProjection(source_kv, target_kv, train_lexicon, test_lexicon, size_src=args.size_src,
                                  size_tgt=args.size_tgt, norm_bf=args.norm_bf)
        scores_MPP = evaluate(lp_MPP.test_matrix_src, lp_MPP.test_matrix_tgt, lp_MPP.W, normalize=True)

        logging.info("Linear projection with Stochastic Gradient Descent")
        lp_SGD = SGD_Projection(source_kv, target_kv, train_lexicon, test_lexicon, size_src=args.size_src,
                                size_tgt=args.size_tgt, norm_bf=args.norm_bf)
        scores_SGD = evaluate(lp_SGD.test_matrix_src, lp_SGD.test_matrix_tgt, lp_SGD.W, normalize=True)

        logging.info("Orthogonal projection")
        lp_ORTH = ORTH_Projection(source_kv, target_kv, train_lexicon, test_lexicon, size_src=args.size_src,
                                  size_tgt=args.size_tgt, norm_bf=args.norm_bf)
        scores_ORTH = evaluate(lp_ORTH.test_matrix_src, lp_ORTH.test_matrix_tgt, lp_ORTH.W, normalize=True)


        print("########################Results############################")
        print("Algebraic Linear Projection:")
        print("Top@1:", scores_MPP[0],"Top@5", scores_MPP[1], "Top@10", scores_MPP[2])
        print("SGD Linear Projection:")
        print("Top@1:", scores_SGD[0], "Top@5", scores_SGD[1], "Top@10", scores_SGD[2])
        print("Orthogonal Projection:")
        print("Top@1:", scores_ORTH[0], "Top@5", scores_ORTH[1], "Top@10", scores_ORTH[2])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description="Word Translations",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--src", default="en", help="Source language")
    parser.add_argument("--tgt", default="es", help="Target language")
    parser.add_argument("--size_src", default=200, type=int, help="Dimension of source embedding")
    parser.add_argument("--size_tgt", default=200, type=int, help="Dimension of target embedding")
    parser.add_argument("--ws", default=10, type=int, help="Window size for learning of keyed vectors")

    parser.add_argument("--create_pre", default=False, type=bool, help="Create preprocessed files")
    parser.add_argument("--create_col", default=False, type=bool, help="Create files that include collocations")
    parser.add_argument("--create_kv", default=False, type=bool,
                        help="Create Keyed Vectors given the configurations else load existing")
    parser.add_argument("--save_kv", default=True, type=bool,
                        help="Save keyed vectors when reuse_kv=False (new creation)")
    parser.add_argument("--norm_bf", default=False, type=bool, help="Normalize embeddings before projection")
    parser.add_argument("--kg", default=False, type=bool, help="Whether to apply on knowledge graph")
    parser.add_argument("--create_lexicon", default=True, type=bool, help="Whether to create language lexicon")
    parser.add_argument("--sub_lexicon", default=False, type=bool, help="Whether to get sublexicon for comparability with KG for languages")
    parser.add_argument("--get_result", default=True, type=bool, help="Whether to launch training/evaluation")

    args = parser.parse_args()

    main()
