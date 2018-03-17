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

    def base_model(self, num_states,X=None,lengths=None):
        if X is None:
            X = self.X
        if lengths is None:
            lengths = self.lengths
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
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

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        Bayesian information criteria: BIC = -2 log L + p log N,
        where L is the likelihood of the fitted model, p is the number of parameters,
        and N is the number of data points. The term -2 log L decreases with
        increasing model complexity (more parameters)

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        min_bic = None
        for states in range(self.min_n_components, self.max_n_components):
            hmm_model = self.base_model(states)
            if hmm_model is None:
                continue
            try:
                logL = hmm_model.score(self.X,self.lengths)
                n_dimensions = self.X.shape[1]
                n_params = (states - 1) + (states * n_dimensions) + states * n_dimensions
                bic = -2 * logL + n_params * math.log(len(self.X))
                if best_model is None or bic < min_bic:
                    best_model = hmm_model
                    min_bic = bic
            except:
                if self.verbose:
                    print("failure to score on {} with {} states".format(self.this_word, states))
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        alpha = 1.0
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        max_DIC = None
        for states in range(self.min_n_components, self.max_n_components):
            hmm_model = self.base_model(states)
            if hmm_model is None:
                continue
            try:
                this_word_Log_L = hmm_model.score(self.X,self.lengths)
                other_words_LogL_sum = 0
                for k,(Xj, j_lengths) in self.hwords.items():
                    if k == self.this_word:
                        continue
                    other_words_LogL_sum += hmm_model.score(Xj,j_lengths)
                anti_evidence = alpha / (len(self.hwords) - 1) * other_words_LogL_sum
                # print("this word log likelihood={}, anti_evidence={}".format(this_word_Log_L,anti_evidence))
                DIC = this_word_Log_L - anti_evidence
                if best_model is None or DIC > max_DIC:
                    best_model = hmm_model
                    max_DIC = DIC
            except:
                if self.verbose:
                    print("failure to score on {} with {} states".format(self.this_word, states))
        return best_model


class SelectorCV(ModelSelector):

    # def cv_model(self, num_states, X_train, train_lengths):
    #     # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=DeprecationWarning)
    #     # warnings.filterwarnings("ignore", category=RuntimeWarning)
    #     try:
    #         hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
    #                                 random_state=self.random_state, verbose=False).fit(X_train, train_lengths)
    #         if self.verbose:
    #             print("model created for {} with {} states".format(self.this_word, num_states))
    #         return hmm_model
    #     except:
    #         if self.verbose:
    #             print("failure on {} with {} states".format(self.this_word, num_states))
    #         return None

    ''' select best model based on average log Likelihood of cross-validation folds    
    '''
    def get_avg_logL(self,split_method,states):
        logL = 0
        folds = 0
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            (X_train, train_lengths) = combine_sequences(cv_train_idx, self.sequences)
            hmm_model = self.base_model(states, np.asarray(X_train), train_lengths)
            if hmm_model is not None:
                (X_test, test_lengths) = combine_sequences(cv_test_idx, self.sequences)
                try:
                    logL += hmm_model.score(X_test,test_lengths) #test_lengths  np.mean(hmm_model._compute_log_likelihood(np.asarray(X_test)))
                    folds += 1
                except:
                    if self.verbose:
                        print("failure to score on {} with {} states".format(self.this_word, states))
            # else:
            #     print('model is none for idx {} and {} states'.format(cv_train_idx, states))

        if folds > 0:
            # print('model created for {} components with {} splits with avg. logL {}.'.format(states, folds, logL/folds))
            return logL / folds
        else:
            # print('could not create model for {} components.'.format(states))
            return float("-inf")

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if len(self.sequences) < 2:
            return None
        n_sp = min(3, len(self.sequences))
        split_method = KFold(n_splits=n_sp)
        # print('{} splits for word {}'.format(n_sp, self.this_word))
        best_states = None
        max_logL = float('-inf')
        for states in range(self.min_n_components, self.max_n_components):
            avg_logL = self.get_avg_logL(split_method,states)
            if avg_logL > max_logL:
                best_states = states

        return self.base_model(best_states)

    # def select(self):
    #     warnings.filterwarnings("ignore", category=DeprecationWarning)
    #     n_sp = min(3, len(self.sequences))
    #     split_method = KFold(n_splits=n_sp)
    #     print('{} splits for word {}'.format(n_sp,self.this_word))
    #     best_model = None
    #     max_logL = 0.0
    #     for states in range(self.min_n_components, self.max_n_components):
    #         logL = 0.0
    #         folds = 0
    #         for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
    #             (X_train, train_lengths) = combine_sequences(cv_train_idx,self.sequences)
    #             hmm_model = self.base_model(states, np.asarray(X_train), train_lengths)
    #             if hmm_model is not None:
    #                 (X_test, test_lengths) = combine_sequences(cv_test_idx, self.sequences)
    #                 logL += np.mean(hmm_model._compute_log_likelihood(np.asarray(X_test)))
    #                 folds += 1
    #             else:
    #                 print('model is none for idx {} and {} states'.format(cv_train_idx,states))
    #
    #         if folds > 0:
    #             print('model created for {} out of {} splits.'.format(folds,n_sp))
    #             logL /= folds
    #
    #         if best_model is None or logL > max_logL:
    #             best_model = hmm_model
    #             max_logL = logL
    #
    #     return best_model

    # def n_zeros_in_matrix(m):
    #     return m.shape[0] * m.shape[1] - np.count_nonzero(m)