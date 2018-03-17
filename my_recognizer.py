import warnings

from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    return recognize_data(models, test_set._data, test_set._hmm_data)


def recognize_data(models: dict, data: dict, hmm_data: dict):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for word_id, sequence in data.items():
        word_probs = {}
        for word, hmm_model in models.items():
            if hmm_model is None:
                word_probs[word] = float("-inf")
            else:
                try:
                    (X, lengths) = hmm_data[word_id]
                    word_probs[word] = hmm_model.score(X,lengths)
                except:
                    word_probs[word] = float("-inf")
        probabilities.append(word_probs)
        bestguess = max(word_probs, key=lambda i: word_probs[i])
        guesses.append(bestguess)
    return probabilities, guesses
