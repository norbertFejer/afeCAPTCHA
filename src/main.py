import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import settings as stt

from sklearn.svm import OneClassSVM
import feature_extractor.fcn as fcn_model

# Global parameters
auc_results = {}
eer_results = {}

##########################################################
################### Data preprocessing ###################
##########################################################

def get_preprocessed_dataset(dataset_path):
    df = pd.read_csv(dataset_path, header=None)

    data_X = df.iloc[:, 0:256].to_numpy().reshape((df.shape[0], 128, 2), order='F')
    data_y = df.iloc[:, 256].to_numpy()
    data_y = data_y - 1
 
    data_y = to_categorical(data_y)

    return data_X, data_y


def get_train_dataset_with_labels_for_extractor_model():
    return get_preprocessed_dataset(stt.TRAIN_DATASET_PATH_EXTRACTOR_MODEL)


####################################################
################### Feature extractor model ###################
####################################################

def get_model_fcn(input_data_shape):
    return fcn_model.Classifier_FCN(stt.OUTPUT_MODEL_DIRECTORY, input_data_shape, stt.NB_CLASSES, stt.VERBOSE)


def get_selected_feature_extractor():
    switcher = {
            0: get_model_fcn
        }

    return switcher.get(stt.SEL_FEATURE_EXTRACTOR.value, lambda: 'Not a valid model name!')


def train_feature_extractor():
    trainX, trainy = get_train_dataset_with_labels_for_extractor_model()

    create_model_func = get_selected_feature_extractor()
    feature_extractor_model = create_model_func( (trainX.shape[1:]) )

    history = feature_extractor_model.fit(trainX, trainy)

    if stt.PLOT_TRAIN:
        fig_name = 'feature_extractor_nb_filters_' + str(fcn_model.stt_fcn.NB_FILTERS)
        plot_train(history, fig_name)


####################################################
################### One Class Classifiers ###################
####################################################

def train_test_occ_with_extracted_features():
    print('\n')
    print("Train dset path: ", stt.TRAIN_DATASET_PATH_OCC_MODEL)
    print("Test dset path: ", stt.TEST_DATASET_PATH_OCC_MODEL)
    print('\n\n')

    df_train = pd.read_csv(stt.TRAIN_DATASET_PATH_OCC_MODEL, header=None)
    df_test = pd.read_csv(stt.TEST_DATASET_PATH_OCC_MODEL, header=None)

    for user_id in range(1, 121):
        username = 'user' + str(user_id)

        train_user_features = get_train_dataset_occ_by_user_id(df_train, user_id)
        print("Train dset shape: ", train_user_features.shape)
        
        test_user_features, test_labels = get_test_dataset_occ_by_user_id(df_test)
        test_user_feature_labels = create_occ_feature_labels(test_labels, user_id)
        print("Test dset shape: ", test_user_features.shape)
        print("Test labels shape: ", test_user_feature_labels.shape)
        print('\n\n')

        train_test_occ_classifier(username, train_user_features, test_user_features, test_user_feature_labels)

    print_result_to_file()


def get_train_dataset_occ_by_user_id(df, user_id):
    # df.iloc[:, nb_filters] means the last column from df, which is the user_id
    nb_filters = fcn_model.stt_fcn.NB_FILTERS
    data_X = df.loc[df.iloc[:, nb_filters] == user_id]
    return data_X.iloc[:, :nb_filters].to_numpy()


def get_test_dataset_occ_by_user_id(df):
    nb_filters = fcn_model.stt_fcn.NB_FILTERS
    return df.iloc[:, 0:nb_filters].to_numpy(), df.iloc[:, nb_filters].to_numpy()


def create_occ_feature_labels(test_labels, user_id):
    return (test_labels == user_id).astype(int)


def train_test_occ_classifier(username, trainX, testX, testy):
    global auc_results
    global agg_block_num

    print('\nTraining model for user:', username, '...')
    # Fit selected network
    classifier = OneClassSVM(kernel='rbf', gamma='scale', verbose=True).fit(trainX)

    print('\nEvaluating model...')
    # Getting AUC from predicted values
    auc_results[username] = []
    eer_results[username] = []
    for i in range(1, stt.AGG_BLOCK_NUM + 1):
        auc, eer = get_auc_eer_metrics(classifier, testX, testy, i)
        auc_results[username].append(auc)
        eer_results[username].append(eer)

    print('Evaluated AUC value for user:', username, 'is', auc_results[username], '\n')
    print('Evaluated AUC value for user:', username, 'is', eer_results[username], '\n')


####################################################
################### Evaluate Model ###################
####################################################

def get_auc_eer_metrics(classifier, testX, y_true, agg_block_num):

    y_pred = classifier.score_samples(testX)
    y_pred = aggregate_blocks(y_pred, agg_block_num)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)

    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return auc, eer


def aggregate_blocks(y_pred, agg_block_num):

    if agg_block_num == 1:
        return y_pred

    for i in range(len(y_pred) - agg_block_num + 1):
        y_pred[i] = np.average(y_pred[i : i + agg_block_num], axis=0)

    return y_pred

####################################################
################### Util Functions ###################
####################################################

def print_result_to_file():
    global auc_results

    file_name = stt.OUTPUT_RESULTS_DIRECTORY + '/ocsvm_f' + str(fcn_model.stt_fcn.NB_FILTERS) + '_auc_res.csv' 
    file = open(file_name, 'w')
    file.write('username,AUC\n')
    
    # Iterating through each user's AUC values
    for user, values in auc_results.items():
        file.write(str(user))
        for value in values:
            file.write(',' + str(value))
        file.write('\n')

    file.close()

    global eer_results

    file_name = stt.OUTPUT_RESULTS_DIRECTORY + '/ocsvm_f' + str(fcn_model.stt_fcn.NB_FILTERS) + '_eer_res.csv' 
    file = open(file_name, 'w')
    file.write('username,EER\n')
    
    # Iterating through each user's EER values
    for user, values in eer_results.items():
        file.write(str(user))
        for value in values:
            file.write(',' + str(value))
        file.write('\n')

    file.close()


def plot_train(history, fig_name):

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    
    plt.title("FCN modell pontossága DFL adathalmaz felhasználóinak tanítása során")
    plt.xlabel("Korszak")
    plt.ylabel("Kategórikus pontosság (categorical accuracy)")
    #plt.xticks()
    plt.yscale('linear')
    #plt.yticks()
    plt.legend(['Tanítás', 'Validálás'], loc='lower right')

    plt.savefig('C:/Anaconda projects/afeCAPTCHA/results/train_plots/' + fig_name + '.png')
    # plt.show()




if __name__ == "__main__":
    # train_feature_extractor()
    train_test_occ_with_extracted_features()