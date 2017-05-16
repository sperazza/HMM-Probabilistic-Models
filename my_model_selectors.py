import math
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold

from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

 Bayesian information criteria: BIC = −2logL + p logN,
  where L is the likelihood of the ﬁtted model, 
  p is the number of parameters, 
  N is the number of data points. 
  The term −2logL decreases with increasing model complexity (more parameters),
  whereas the penalties 2p or p logN increase with increasing complexity. 
  The BIC applies a larger penalty when N > e2 = 7.
 
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float("Inf")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                log_l = model.score(self.X, self.lengths)
                p = n ** 2 + 2 * n * len(self.X[0]) - 1
                score = (-2) * log_l + math.log(len(self.X)) * p
                if score < best_score:
                    best_score, best_model = score, model
            except:
                pass
        return best_model


class SelectorDIC(ModelSelector):
    """ select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    
    from forum:https://discussions.udacity.com/t/dic-criteria-clarification-of-understanding/233161
    You should find the number of components where the difference is largest. 
    The idea of DIC is that we are trying to find the model that gives a high likelihood(small negative number)
     to the original word and low likelihood(very big negative number) to the other words. So DIC score is
    DIC = log(P(original world)) - average(log(P(otherwords)))
   As you can see from the formula, lower the P for the other words,higher -log and DIC score. So we are looking for max DIC score.
    """

    def dic_score(self, n):
        model = self.base_model(n)
        other_scores = []
        for word, x_len in self.hwords.items():
            if word is not self.this_word:
                other_scores.append(model.score(*x_len))
        return model.score(self.X, self.lengths) - np.mean(other_scores), model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores
        best_score = float("-Inf")
        best_model = None
        try:
            for n in range(self.min_n_components, self.max_n_components + 1):
                score, model = self.dic_score(n)
                if score > best_score:
                    best_score = score
                    best_model = model
        except:
            pass
        return best_model


class SelectorCV(ModelSelector):
    """ select best model based on average log Likelihood of cross-validation folds

    """

    # TODO implement model selection using CV
    def cv_score(self, n):
        scores = []
        split_method = KFold(n_splits=2)
        model = self.base_model(n)
        for train_idx, test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(train_idx, self.sequences)
            model = self.base_model(n)
            x_l = combine_sequences(test_idx, self.sequences)
            scores.append(model.score(*x_l))
        mean_scores = np.mean(scores)
        return mean_scores, model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float("Inf")
        best_model = None
        try:
            for n in range(self.min_n_components, self.max_n_components + 1):
                score, model = self.cv_score(n)
                if score < best_score:
                    best_score = score
                    best_model = model
        except:
            pass
        if best_model is None:
            return self.base_model(self.n_constant)
        return best_model
