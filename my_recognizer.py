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
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for test_word, x_len in test_set.get_all_Xlengths().items():
        best_score = float("-inf")
        best_guess = ""
        p_dict = {}
        for train_word, model in models.items():
            try:
                log_p = model.score(*x_len)
                p_dict[train_word] = log_p
            except:
                p_dict[train_word] = float("-inf")
            if log_p > best_score:
                best_score = log_p
                best_guess = train_word
        probabilities.append(p_dict)
        guesses.append(best_guess)
        # sorted(probability_dict,key:p_dict.keys())
    return probabilities, guesses
