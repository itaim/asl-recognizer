import warnings
import numpy as np
import math
import pickle
from hmmlearn.hmm import GaussianHMM
from matplotlib import (cm, pyplot as plt, mlab)
import arpa
from asl_utils import show_errors
from my_data_preparation import asl, features_ground, features_norm, features_polar, features_delta,features_custom
from my_model_selectors import SelectorBIC, SelectorDIC, SelectorCV
from my_recognizer import recognize
import re
import itertools


def train_a_word(word, num_hidden_states, features):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))
    variance = np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()

def demo_train():
    demoword = 'BOOK'
    model, logL = train_a_word(demoword, 3, features_ground)
    # print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
    # print("logL = {}".format(logL))
    # show_model_stats(demoword, model)

# demo_train()

def visualize(word, model):
    """ visualize the input model for a particular word """
    variance = np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    figures = []
    for parm_idx in range(len(model.means_[0])):
        xmin = int(min(model.means_[:, parm_idx]) - max(variance[:, parm_idx]))
        xmax = int(max(model.means_[:, parm_idx]) + max(variance[:, parm_idx]))
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i, parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()


def demo_visualize():
    my_testword = 'CHOCOLATE'
    model, logL = train_a_word(my_testword, 3, features_ground) # Experiment here with different parameters
    show_model_stats(my_testword, model)
    print("logL = {}".format(logL))
    visualize(my_testword, model)

def train_all_words2(training, model_selector):
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict


def grid_search_models():
    model_selectors = {"CV":SelectorCV, "BIC":SelectorBIC, "DIC":SelectorDIC}
    features_dict = {"GROUND":features_ground, "NORM":features_norm, "POLAR":features_polar,"DELTA":features_delta,"CUSTOM":features_norm}
    wers = {}
    grid = {}
    for feat_name, features in features_dict.items():
        test_set = asl.build_test(features)
        words_data = asl.build_training(features)
        for sel_name,selector in model_selectors.items():
            models = train_all_words2(words_data, selector)
            probabilities, guesses = recognize(models, test_set)
            key = sel_name + "-" + feat_name
            print(key)
            wer = show_errors(guesses, test_set)
            wers[key] = wer
            grid[key] = (wer,models)
    s = [(k,wers[k]) for k in sorted(wers,key = wers.get,reverse=False)]
    print(s)
    pickle.dump(grid, open("data/grid_models.pkl", "wb"))

# def avg_sequences_probs(features_sequence_prob):
#     print("{} {}".format(len(features_sequence_prob),type(features_sequence_prob)))
#     d = {}
#     for seq_probs in features_sequence_prob:
#         for word in seq_probs[0].iterkeys():
#             d[word] = avg_models_scores([d[word] for d in seq_probs])
#     return d


#polar 0.52
#ground 0.54
#delta-rx 0.6
#DIC['norm 0.64
def avg_models_scores(scores):
    sum_w = sum([w for (s,w) in scores if s > float("-inf")])
    s = [s*w/sum_w for (s,w) in scores if s > float("-inf")]
    if len(s) < len(scores)//2 or sum_w < 0.5:
        return float("-inf")
    else:
        return sum(s)/len(s)


def merge_seq_dicts(ground,norm,polar,delta,custom,features_weights):
    ds = [(ground,features_weights["ground"]),(norm,features_weights["norm"]),(polar,features_weights["polar"]),(delta,features_weights["delta"]),(delta,features_weights["custom"])]
    d = {}
    for k in ground.keys():
        d[k] = avg_models_scores(list((d[k],w) for (d,w) in ds))
    return d


def get_prob(combination):
    visual_model_score =  sum([p for (w,p) in combination])/len(combination)  #reduce(lambda x,y:x*y,)
    sentence = ' '.join([re.sub("\d+$", "", w) for (w, p) in combination])
    alpha = 0.4
    beta = 1 - alpha
    return alpha * language_model.log_s(sentence) + beta * visual_model_score


def find_best_match(sentence_options):
    combinations = list(itertools.product(*sentence_options))
    return max(combinations,key=lambda c:get_prob(c))


combinations_per_sentence = 750

def guess_by_combination(test_set, ensemble_probabilities):
    guesses = []
    for video_num, indices in test_set.sentences_index.items():
        sentence_options = []
        options_per_word = round(combinations_per_sentence ** (1/len(indices)))
        for i in indices:
            word_probs = ensemble_probabilities[i].items()
            t = min(options_per_word, len(word_probs) - 1)
            # print("list len {}, taking {}".format(len(l), t))
            top_word_matches = sorted(word_probs, key=lambda x: x[1], reverse=True)[0:t]
            # print("top word matches {}".format(top_word_matches))
            sentence_options.append(top_word_matches)
        guess = find_best_match(sentence_options)
        # print("best guess {} for video {}".format(guess,video_num))
        guesses.extend(guess)

    return guesses


def ensemble_models():
    features_data = [(name, f, s) for (name, f, s) in
                    [("ground", features_ground, SelectorBIC), ("norm", features_norm,SelectorDIC), ("polar", features_polar,SelectorBIC),
                     ("delta", features_delta,SelectorCV), ("custom", features_custom,SelectorDIC)]]
    features_probs = {}
    features_success_rates = {}
    test_set = None
    for name, features, selector in features_data:
        test_set = asl.build_test(features)
        words_data = asl.build_training(features)
        models = train_all_words2(words_data, selector)
        #probabilities is a list of dictionaries, each dictionary gives each possible word prob for the sequence [{word:prob}]
        probabilities, guesses = recognize(models, test_set)
        #todo use the feature relative wer as its proportion in the ensemble output
        features_probs[name] = probabilities
        print("{} features:".format(name))
        wer = show_errors(guesses, test_set)
        features_success_rates[name] = 1-wer

    sm = sum(features_success_rates.values())
    features_weights = [(k,v/sm) for (k,v) in features_success_rates.items()]
    pickle.dump({"features_probs":features_probs,"features_weights":features_weights}, open("data/feature_models_data.pkl","wb"))

    ensemble_guess(features_probs,features_weights,test_set)

    #[avg_sequences_probs([ground,norm,polar,delta]) for (ground,norm,polar,delta) in list(zip(*model_probs.values()))]

language_model = arpa.loadf("lm/ukn.3.lm")[0]


def ensemble_guess(features_probs=None,features_weights=None,test_set=None):
    if features_probs is None or features_weights is None:
        l = pickle.load(open("data/feature_models_data.pkl","rb"))
        features_probs = l["features_probs"]
        features_weights = l["features_weights"]
        print("feature models data loaded")

    if test_set is None:
        test_set  =  asl.build_test(features_ground)

    features_weights = dict(features_weights)
    ensemble_probabilities = [merge_seq_dicts(ground, norm, polar, delta, custom, features_weights) for
                              (ground, norm, polar, delta, custom)
                              in zip(features_probs["ground"], features_probs["norm"], features_probs["polar"],
                                     features_probs["delta"], features_probs["custom"])]

    pguesses = guess_by_combination(test_set, ensemble_probabilities)
    guesses = [w for (w,p) in pguesses]
    # guesses = []
    # for word_probs in ensemble_probabilities:
    #     bestguess = max(word_probs, key=lambda i: word_probs[i])
    #     guesses.append(bestguess)
    # print('Ensembled Guess')
    show_errors(guesses, test_set)


def guess_with_cv_polar_and_slm():
    training = asl.build_training(features_polar)
    models_cv_polar = train_all_words2(training, SelectorCV)
    test_set = asl.build_test(features_polar)
    probabilities, guesses = recognize(models_cv_polar, test_set)
    print("With cv_polar models")
    show_errors(guesses, test_set)

    pguesses = guess_by_combination(test_set, probabilities)
    guesses = [w for (w, p) in pguesses]
    print("With cv_polar models and language model")
    show_errors(guesses, test_set)


# guess_with_cv_polar_and_slm()
# ensemble_models()
ensemble_guess()
# grid_search_models()
