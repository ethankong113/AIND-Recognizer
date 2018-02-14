import warnings
from asl_data import SinglesData

def get_best_word(log_likelihoods):
    best_word = None
    best_score = 0
    for word, score in log_likelihoods.items():
        if best_word == None or score > best_score:
            best_word, best_score = word, score
    return best_word

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
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for word_id, val in test_set.get_all_Xlengths().items():
        current_sequence, current_lengths = test_set.get_item_Xlengths(word_id)
        log_likelihoods = {}
        for word, model in models.items():
            try:
                LogLvalue = model.score(current_sequence, current_lengths)
                log_likelihoods[word] = LogLvalue
            except:
                log_likelihoods[word] = float("-inf")
                continue
        probabilities.append(log_likelihoods)
        guesses.append(get_best_word(log_likelihoods))
    return probabilities, guesses



    # for id in range(0, len(test_set.get_all_Xlengths()))
