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
    ''' select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf

    BIC = -2 * logL + p * logN
    '''

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        best = (None, float('inf')) # Tuple (model, BIC score)

        for n in range(self.min_n_components, self.max_n_components):
            try:
                # Train HMM
                model = GaussianHMM(n_components=n, n_iter=1000,
                                    random_state=self.random_state).fit(self.X, self.lengths)
    
                logL = model.score(self.X, self.lengths)
                logN = np.log(len((self.lengths))) # N is number of data points
                p = n ** 2 + 2 * n * model.n_features - 1 # p is number of parameters
    
                # Calculate BIC (Bayesian Information Criteria) score      
                score = -2 * logL + p * logN
                # If BIC score is better than previous best, store model and the score
                if score < best[1]:
                    best = model, score
            except:
               pass
           
        return best[0]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf

    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best = (None, float('-inf')) # Tuple (model, DIC score)

        for n in range(self.min_n_components, self.max_n_components):
            try:
                # Train HMM
                model = GaussianHMM(n_components=n, n_iter=1000,
                                    random_state=self.random_state).fit(self.X, self.lengths)
    
                M = len((self.words).keys()) # Number of words
                logL = model.score(self.X, self.lengths)
                
                # REMINDER: X is a numpy array of feature lists; lengths is
                # a list of lengths of sequences within X; should always have only one item in lengths
                log_sum = 0
                for word in self.words.keys():
                    X, lengths = self.hwords[word]
                    log_sum += model.score(X, lengths)
                                   
                # Calculate DIC (Discriminative Information Criterion) score      
                score = logL - (1 / (M - 1)) * (log_sum - logL)
                # If DIC score is better than previous best, store model and the score
                if score > best[1]:
                    best = model, score
            except:
               pass
           
        return best[0]

class SelectorCV(ModelSelector):
    ''' select best model based on average Log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best = (None, float('-inf')) # Tuple (model, score)

        k_fold = KFold() # 3 folds by default

        # Fit series of HMM with n hidden states in a range from min_n_components to max_n_components
        for n in range(self.min_n_components, self.max_n_components):
            
            scores = []        
            
            if len(self.sequences) > 2: # Check if available samples is sufficient for 3-fold cross-validation
                for indexes in k_fold.split(self.sequences): # For each fold
                    # Re-combining sequences split
                    training_values, training_lengths = combine_sequences(indexes[0], self.sequences)
                    testing_values, testing_lengths = combine_sequences(indexes[1], self.sequences)

                    try:
                        # Training HMM
                        model = GaussianHMM(n_components=n, n_iter=1000,
                                            random_state=self.random_state).fit(training_values, training_lengths)
                        # Testing HMM, storing Log Likelihood score to the list             
                        scores.append(model.score(testing_values, testing_lengths))
                    except:
                        pass
            else: # Skip doing k-folds CV
                try: # X3 ;)
                    model = GaussianHMM(n_components=n, n_iter=1000,
                                        random_state=self.random_state).fit(self.X, self.lengths)
                    scores.append(model.score(self.X, self.lengths))
                except:
                    pass
            
            if scores:
                if np.mean(scores) > best[1]:
                    best = model, np.mean(scores)
           
        return best[0]