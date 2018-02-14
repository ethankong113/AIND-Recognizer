import math
import statistics
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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    """
    Sources:
    [2] https://stats.stackexchange.com/questions/79955/scikit-learn-gaussianhmm-decode-vs-score
    """

    def calc_num_of_params(self, num_of_states, num_of_dp):
        """
        number of params is also number of free params
        Sources:
        https://discussions.udacity.com/t/understanding-better-model-selection/232987/9
        """
        return (num_of_states ** 2) + (2 * num_of_states * num_of_dp) - 1


    def calc_bic_score(self, log_likelihood, num_of_params, log_num_of_dp):
        return (-2 * log_likelihood) + (num_of_params * log_num_of_dp)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states)
                log_likelihood = hmm_model.score(self.X, self.lengths)
                num_of_dp = sum(self.lengths)
                log_num_of_dp = np.log(num_of_dp)
                num_of_params = self.calc_num_of_params(num_states, num_of_dp)
                bic_score = self.calc_bic_score(log_likelihood, num_of_params, log_num_of_dp)
                models.append((bic_score, hmm_model))
            except:
                pass
        if not len(models):
            return None

        return min(models, key=lambda model: model[0])[1]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def calc_dic_score(self, log_likelihood_this, hmm_model, other_words):
        avg_log_likelihood_others =  np.mean([hmm_model.score(word[0], word[1]) for word in other_words])
        return log_likelihood_this - avg_log_likelihood_others

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        other_words = []
        models = []

        [other_words.append(self.hwords[word]) for word in self.words if word != self.this_word]

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states)
                log_likelihood_this = hmm_model.score(self.X, self.lengths)
                dic_score = self.calc_dic_score(log_likelihood_this, hmm_model, other_words)
                models.append((dic_score, hmm_model))
            except:
                pass

        if not len(models):
            return None

        return max(models, key=lambda model: model[0])[1]



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        kf = KFold(3)
        models = []
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                if len(self.sequences) > 2:
                    cvs_scores = []
                    hmm_model = self.base_model(num_states)
                    for train_index, test_index in kf.split(self.sequences):
                        X_test, lengths_test = combine_sequences(test_index, self.sequences)
                        log_likelihood = hmm_model.score(X_test, lengths_test)
                        cvs_scores.append(log_likelihood)
                    avg_score = np.mean(cvs_scores)
                    models.append((avg_score, hmm_model))

                else:
                    hmm_model = self.base_model(num_states)
                    log_likelihood = hmm_model.score(X, lengths)
                    models.append((log_likelihood, hmm_model))
            except:
                pass

        if not len(models):
            return self.base_model(self.min_n_components)

        return max(models, key=lambda model: model[0])[1]
