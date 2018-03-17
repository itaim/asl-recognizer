from my_data_preparation import *
from asl_utils import combine_sequences
from sklearn.model_selection import KFold
from my_model_selectors import SelectorCV, SelectorDIC, SelectorBIC

# train a hmm with CV for each feature set and leave on out of fold
# score oof with hmms
from my_model_selectors import SelectorCV
from my_recognizer import recognize


def get_train_test_folds(sequences):
    n_sp = min(3, len(sequences))
    split_method = KFold(n_splits=n_sp)
    train_test_splits = []
    for cv_train_idx, cv_test_idx in split_method.split(sequences):
        train = combine_sequences(cv_train_idx, sequences)
        test = combine_sequences(cv_test_idx, sequences)
        train_test_splits.append((train, test))
    return train_test_splits

def train_ensemble_hmm_for_all_words():
    all_features = [(name, f, asl.asl.build_training(f)) for (name,f) in
                    [("ground_state",features_ground), ("norm_state",features_norm), ("polar_state",features_polar), ("delta_state",features_delta)]]

    new_data = {}
    new_hmm_data = {}
    for word in all_features[0][1].words:
        new_data[word] = []
        for name, features, training in all_features:
            all_sequences = training.get_all_sequences()
            all_lengths = training.get_all_Xlengths()
            sequences = all_sequences[word]
            train_test_splits = get_train_test_folds(sequences)
            feature_model = SelectorCV(sequences, all_lengths, word,
                               n_constant=3).select_with_oof(train_test_splits[:, -1])
            #use the feature model state prediction as a feature in new space and train an "ensemble" hmm on it
            (X_train, train_lengths), _ = train_test_splits[-1]
            _, state_sequence = feature_model.decode(np.asarray(X_train), np.asarray(train_lengths), algorithm="viterbi")
            #todo add the state as feature for each frame in asl.df X_train indices
            # add feature column with state values at right indices
            asl.df.iloc[1,asl.df.columns.get_loc(name)] = state_sequence

    #train the ensemble hmm for the word on the left out fold new feature space
    models_dict = {}
    for word in new_data.keys():
        ensemble_model = SelectorDIC(new_data, new_hmm_data, word,
                                   n_constant=3).select()
        models_dict[word] = ensemble_model


        model_dict[word]=(feature_model, train_test_splits[-1])
    return model_dict



def ensemble_hmm():
    models = train_ensemble_hmm_for_all_words()
    test_set = asl.build_test(features)
    probabilities, guesses = recognize(models, test_set)
    show_errors(guesses, test_set)

            probabilities, guesses = recognize(word_models, test_set)
            key = sel_name + "-" + feat_name
            wer = show_errors(guesses, test_set)
            wers[key] = wer
            grid[key] = (wer, word_models)

    selector = SelectorCV()
    selector.select_with_oof()

