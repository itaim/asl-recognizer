import arpa
import re
import pickle
import pandas as pd
import itertools
from functools import reduce

import numpy as np

from asl_data import SinglesData
from asl_utils import show_errors

ukn3 = arpa.loadf("lm/ukn.3.lm")
lm = ukn3[0]

probabilities = pickle.load(open("data/probabilities.pkl","rb"))
test_set = pickle.load(open("data/test_set.pkl","rb"))
df_probs = pd.DataFrame(probabilities)
# print(df_prob.head())
lm_factor = 20.0

def score_with_lm1():
    for video_num, indices in test_set.sentences_index.items():
        #     visual_model_guesses = df_probs.iloc[indices,:].idxmax(axis=1)
        ngram_indices = []
        for sentence_idx, word_idx in enumerate(indices):
            if ngram_indices:
                ngram_prefix = df_probs.iloc[ngram_indices, :].idxmax(axis=1).tolist()
                row = df_probs.iloc[word_idx, :]
                for col_idx, col in enumerate(row.iteritems()):
                    word = re.sub("\d+$", "", col[0])
                    ngram = ngram_prefix + [word]
                    visual_model_log_l = col[1]
                    ngram_log_l = lm.log_p(ngram)
                    updated_likelihood = lm_factor * (1 / len(ngram_prefix)) * ngram_log_l + visual_model_log_l
                    df_probs.iloc[word_idx, col_idx] = updated_likelihood
                    print("word {} ngram {} ngram_L={} visual_model_L={} updated L = {}".format(word,ngram,ngram_log_l,visual_model_log_l,updated_likelihood))

            ngram_indices.append(word_idx)
            if len(ngram_indices) > 2:
                ngram_indices.pop(0)
    guesses = df_probs.idxmax(axis=1).tolist()
    print("guesses len {}, test set wordlist len {}".format(len(guesses), len(test_set.wordlist)))
    # print("{}".format(guesses))
    show_errors(guesses, test_set)

top_mathces_to_consider = 5


def get_prob(combination):
    visual_model_score =  sum([p for (w,p) in combination])/len(combination)  #reduce(lambda x,y:x*y,)
    sentence = ' '.join([re.sub("\d+$", "", w) for (w, p) in combination])
    # language_model_score = lm.log_s(sentence)
    # score = lm_factor * language_model_score + visual_model_score
    # if sentence == "JOHN WRITE HOMEWORK":
    #     print("sentence {}, vm score={}, lm score={}, score={}".format(sentence, visual_model_score, language_model_score,score))
    # print("sentence {}, vm score={}, lm score={}, score={}".format(sentence,visual_model_score,language_model_score,score))
    # return score
    return lm.log_s(sentence)


def find_best_match(sentence_options):
    combinations = list(itertools.product(*sentence_options))
    return max(combinations,key=lambda c:get_prob(c))


combinations_per_sentence = 750

def guess_by_combination():
    guesses = []
    for video_num, indices in test_set.sentences_index.items():
        # if video_num != 2:
        #     continue
        sentence_options = []
        options_per_word = round(combinations_per_sentence ** (1/len(indices)))
        for i in indices:
            top_word_matches = df_probs.iloc[i, :].sort_values(ascending=False).head(options_per_word).iteritems()
            sentence_options.append(top_word_matches)
        guess = find_best_match(sentence_options)
        print("best guess {} for video {}".format(guess,video_num))
        guesses.extend(guess)

    return guesses

def analyze_guesses(guesses: list, test_set: SinglesData):

    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({}!={})!".format(len(guesses), num_test_words))
    for word_id in range(num_test_words):
        correct = test_set.wordlist[word_id]
        guess = guesses[word_id]
        guess_prob = df_probs.iloc[word_id, df_probs.columns.get_loc(guess)]
        correct_prob = df_probs.iloc[word_id, df_probs.columns.get_loc(correct)]
        top_5_matches = df_probs.iloc[word_id, :].sort_values(ascending=False).head(5).iteritems()
        print("guess {} probability {}, correct {} probability {}".format(guess, guess_prob, correct, correct_prob))
        for m in top_5_matches:
            print("match: {}".format(m))
        # print(top_5_matches)
        if guess != correct:
            S += 1
    WER = float(S) / float(N)
    print("\n**** WER = {}".format(WER))
    print("Total correct: {} out of {}".format(N - S, N))
    print('Video  Recognized                                                    Correct')
    print('=====================================================================================================')
    for video_num in test_set.sentences_index:
        correct_sentence = [test_set.wordlist[i] for i in test_set.sentences_index[video_num]]
        recognized_sentence = [guesses[i] for i in test_set.sentences_index[video_num]]
        for i in range(len(recognized_sentence)):
            if recognized_sentence[i] != correct_sentence[i]:
                recognized_sentence[i] = '*' + recognized_sentence[i]
        print('{:5}: {:60}  {}'.format(video_num, ' '.join(recognized_sentence), ' '.join(correct_sentence)))
    return WER


# guesses = score_with_lm2()
# show_errors(guesses, test_set)

analyze_guesses(df_probs.idxmax(axis=1).tolist(), test_set)
# score_with_lm1()