from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from asl_data import create_hmmlearn_data
from my_training import *

# Algorithm 1 The Segmentally-Boosted Hidden Markov Models (SBHMMs) Algorithm
# 1: Train HMMs by Expectation Maximization (EM) using the time sequence training data.
# 2: Find the optimal state transition path by the Viterbi decoding algorithm.
# 3: Label every sample with its most likely hidden state.
# 4: Train AdaBoost ensembles for this labeling.
# 5: Project the data to a new feature space using the ensembles.
# 6: Train HMMs by EM in the new feature space.
# 7: In testing, project the test data to the same new feature space and predict their label
# using the HMMs computed in Step 6.


# Note, since the EM algorithm is a gradient-based optimization method,
# it will generally get stuck in local optima.

def train_hmms(words_data, model_selector):
    sequences = words_data.get_all_sequences()
    Xlengths = words_data.get_all_Xlengths()
    num_features =  len(next(iter(sequences.values()))[0][0])
    mapping_observation_state = np.empty((0, num_features + 1))
    for word in words_data.words:
        model = model_selector(sequences, Xlengths, word,
                               n_constant=3).select()
        X, lengths = Xlengths[word]
        word_index = words_data.words.index(word)
        try:
            logp, y = model.decode(X, lengths)
            # print("decoded {}".format(np.array_str(y.reshape(-1, 1))))
            y += (1000 * word_index)
        except Exception as e:
            print(e)
            print("model is none or predict error for word {} with {} sequences".format(word, X.shape[0]))
            y = np.zeros(X.shape[0])
            y += (1000 * word_index)

        a = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        mapping_observation_state = np.concatenate((mapping_observation_state, a))
    return mapping_observation_state

n_estimators = 600

def train_adaboost_ensembles(X,y):
    ensemble_list = []
    le = LabelEncoder()
    y_new = le.fit_transform(y)
    for class_label in le.classes_:
        y_class_label = y_new == le.transform([class_label])
        y_class_label = y_class_label.astype(int)  # hack to convert boolean array into numeric array
        ensemble = AdaBoostClassifier(n_estimators=n_estimators, base_estimator=DecisionTreeClassifier(max_depth=2), learning_rate=1)
        ensemble.fit(X, y_class_label)
        ensemble_list.append(ensemble)
    return ensemble_list


def ensemble_scores(ensemble_list, X):
    scores = []  #np.zeros((X.shape[0], len(ensemble_list)))
    for i, ensemble in enumerate(ensemble_list):
        scores.append(ensemble.decision_function(X).tolist())
    return scores


def project_data_to_new_feature_space(ensembles, data):
    new_data = {}
    # data = {"BOOK":data["BOOK"]}
    for key, sequences in data.items():
        # print("getting ensembles scores for key {} with {} sequences".format(key,len(sequences)))
        new_data[key] = []
        for sequence in sequences:
            scores = ensemble_scores(ensembles, np.asarray(sequence))
            # print("adding ensemble_scores shape {} from sequence of {}".format(scores.shape, len(sequence)))
            new_data[key].append(scores)
    return new_data


def train_sbhmms(new_sequences, model_selector):
    new_XLengths = create_hmmlearn_data(new_sequences)
    model_dict = {}
    for word in new_sequences.keys():
        model = model_selector(new_sequences, new_XLengths, word,
                               n_constant=3).select()
        model_dict[word] = model
    return model_dict


def sbhmms():
    words_data = asl.build_training(features_polar)
    print("training HMMs and adding labeling sequences with predictions")
    labeled_data = train_hmms(words_data, SelectorDIC)
    print("training adaboost ensembles on state transition labeled data")
    ensembles = train_adaboost_ensembles(labeled_data[:,:-1], labeled_data[:,-1])
    pickle.dump(ensembles, open("data/sbhmm_ensembles.pkl", "wb"))
    part2(words_data,ensembles)

def part2(words_data = None,ensembles = None):
    if not ensembles:
        ensembles = pickle.load(open("data/sbhmm_ensembles.pkl", "rb"))
        print("ensembles loaded from pickle dump")
    if not words_data:
        words_data = asl.build_training(features_polar)
    new_sequences = project_data_to_new_feature_space(ensembles, words_data._data)
    pickle.dump(new_sequences, open("data/projected_data.pkl", "wb"))
    part3(new_sequences,ensembles)


def part3(new_sequences=None, ensembles=None):
    if not new_sequences:
        new_sequences = pickle.load(open("data/projected_data.pkl", "rb"))
        print("ensembles projected data loaded from pickle dump")
    if not ensembles:
        ensembles = pickle.load(open("data/sbhmm_ensembles.pkl", "rb"))
        print("ensembles loaded from pickle dump")

    print("training SBHMMs on data projected to ensembles feature space")
    sbhmm_models = train_sbhmms(new_sequences, SelectorDIC)
    pickle.dump({"models": sbhmm_models, "ensembles": ensembles}, open("data/sbhmm_models.pkl", "wb"))
    part4(ensembles,sbhmm_models)

def part4(ensembles = None,sbhmm_models= None):
    if not ensembles or not sbhmm_models:
        d = pickle.load(open("data/sbhmm_models.pkl", "rb"))
        ensembles = d["ensembles"]
        sbhmm_models = d["models"]
        print("ensembles and sbhmm models loaded from pickle dump")

    test_set = asl.build_test(features_polar)
    print("projecting test data to new feature space")
    new_sequences = project_data_to_new_feature_space(ensembles, test_set._data)
    new_XLengths = create_hmmlearn_data(new_sequences)
    print("using SBHMMs to guess test data")
    probabilities, guesses = recognize(sbhmm_models, new_sequences, new_XLengths)
    wer = show_errors(guesses, test_set)

# part2()
sbhmms()

# part4()