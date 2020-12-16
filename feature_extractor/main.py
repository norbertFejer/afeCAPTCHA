import pandas as pd
import numpy as np
import fcn as fcn_model

from keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import seaborn as sns

# train_dataset_path = 'c:/Anaconda projects/afeCAPTCHA/datasets/preprocessed_dataset/sapimouse_3min.csv'
# test_dataset_path = 'c:/Anaconda projects/afeCAPTCHA/datasets/preprocessed_dataset/sapimouse_1min.csv'

train_dataset_path = 'c:/Anaconda projects/Software_mod/analysis/raw_data/sapimouse_128_3min.csv'
test_dataset_path = 'c:/Anaconda projects/Software_mod/analysis/raw_data/sapimouse_128_1min.csv'

output_model_directory = 'C:/Anaconda projects/afeCAPTCHA/trained_models'
output_results_directory = 'C:/Anaconda projects/afeCAPTCHA/results'
nb_filters = 128
fcn_model_path = 'C:/Anaconda projects/afeCAPTCHA/trained_models/best_fcn_model_nb_f' + str(nb_filters) + '.hdf5'
nb_classes = 123
agg_block_num = 5
verbose = True

# Global parameters
auc_results = {}
eer_results = {}



def train_fcn(nb_filters=128):
    trainX, trainy = get_raw_dataset(train_dataset_path)

    trainX = scale_dataset(trainX)

    fcn = fcn_model.Classifier_FCN(output_model_directory, (trainX.shape[1:]), nb_classes, nb_filters, verbose)
    history = fcn.fit(trainX, trainy)

    fig_name = 'fig_nb_filters_' + str(nb_filters)
    plot_train(history, fig_name)


def get_raw_dataset(dataset_path):
    df = pd.read_csv(dataset_path, header=None)

    data_X = df.iloc[:, 0:256].to_numpy().reshape((df.shape[0], 128, 2), order='F')
    data_y = df.iloc[:, 256].to_numpy()
    data_y = data_y - 1
 
    data_y = to_categorical(data_y)

    return data_X, data_y


def get_feature_extractor(model_path):

    print('Loaded model path:', model_path)

    model = load_model(model_path)
    model._layers.pop()
    model.outputs = [model.layers[-1].output]

    return model


def train_test_ocsvm_classifier(username, trainX, testX, testy):
    global auc_results
    global agg_block_num

    print('\nTraining model for user:', username, '...')
    # Fit selected network
    classifier = OneClassSVM(kernel='rbf', gamma='scale', verbose=True).fit(trainX)

    print('\nEvaluating model...')
    # Getting AUC from predicted values
    auc_results[username] = []
    eer_results[username] = []
    for i in range(1, agg_block_num+1):
        auc, eer = get_auc_eer_metrics(classifier, testX, testy, i)
        auc_results[username].append(auc)
        eer_results[username].append(eer)

    print('Evaluated AUC value for user:', username, 'is', auc_results[username], '\n')
    print('Evaluated AUC value for user:', username, 'is', eer_results[username], '\n')


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



def scale_dataset(data):

    for i in range(data.shape[0]):
        data[i] = standard_scaler(data[i])

    return data


def standard_scaler(data):

    mean_val = np.mean(data, axis=0)
    std_val = np.std(data, axis=0)

    if std_val[0] == 0:
        std_val[0] = 0.001
    if std_val[1] == 0:
        std_val[1] = 0.001

    data = (data - mean_val) / std_val

    return data


def get_features_from_model(model, raw_data):
    return model.predict(raw_data)


def get_train_dataset_ocsvm_by_user_id(df, user_id):
    data_X = df.loc[df.iloc[:, nb_filters] == user_id]
    return data_X.iloc[:, :nb_filters].to_numpy()


def get_test_dataset_ocsvm_by_user_id(df):
    return df.iloc[:, 0:nb_filters].to_numpy(), df.iloc[:, nb_filters].to_numpy()


def get_feature_labels(test_labels, user_id):
    return (test_labels == user_id).astype(int)


def train_test_ocsvm():
    df_train = pd.read_csv(train_dataset_path, header=None)
    df_test = pd.read_csv(test_dataset_path, header=None)

    feature_extractor_model = get_feature_extractor(fcn_model_path)

    for user_id in range(1, 124):
        username = 'user' + str(user_id)

        train_raw_data = get_train_dataset_ocsvm_by_user_id(df_train, user_id)
        train_features = get_features_from_model(feature_extractor_model, train_raw_data)
        
        test_raw_data, test_labels = get_test_dataset_ocsvm_by_user_id(df_test)
        test_features = get_features_from_model(feature_extractor_model, test_raw_data)
        test_feature_labels = get_feature_labels(test_labels, user_id)

        train_test_ocsvm_classifier(username, train_features, test_features, test_feature_labels)

    print_result_to_file()


def print_result_to_file():
    """ Save evaluation results to file

        Parameters:
            file_name (str) - filename

        Returns:
            None
    """
    global auc_results

    file_name = output_results_directory + '/ocsvm_f' + str(nb_filters) + '_auc_res.csv' 
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

    file_name = output_results_directory + '/ocsvm_f' + str(nb_filters) + '_eer_res.csv' 
    file = open(file_name, 'w')
    file.write('username,EER\n')
    
    # Iterating through each user's EER values
    for user, values in eer_results.items():
        file.write(str(user))
        for value in values:
            file.write(',' + str(value))
        file.write('\n')

    file.close()

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


def train_test_ocsvm_with_predefined_features(train_dataset_path, test_dataset_path):
    print('\n')
    print("Train dset path: ", train_dataset_path)
    print("Test dset path: ", test_dataset_path)
    print('\n\n')

    df_train = pd.read_csv(train_dataset_path, header=None)
    df_test = pd.read_csv(test_dataset_path, header=None)

    for user_id in range(1, 124):
        username = 'user' + str(user_id)

        train_user_features = get_train_dataset_ocsvm_by_user_id(df_train, user_id)
        print("Train dset shape: ", train_user_features.shape)
        
        test_user_features, test_labels = get_test_dataset_ocsvm_by_user_id(df_test)
        test_user_feature_labels = get_feature_labels(test_labels, user_id)
        print("Test dset shape: ", test_user_features.shape)
        print("Test labels shape: ", test_user_feature_labels.shape)
        print('\n\n')

        train_test_ocsvm_classifier(username, train_user_features, test_user_features, test_user_feature_labels)

    print_result_to_file()


########################################################################################################################

def plot_results_boxplot():
    df = pd.read_csv("C:/Anaconda projects/afeCAPTCHA/results/fin_res.csv")

    sns.boxplot(data=df, linewidth=2)

    plt.title('SapiMouse: 123 felhasználó', fontsize=30)
    plt.ylabel('AUC érték', fontsize=30)
    #plt.xlabel('SapiMouse 1min', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=28)
    plt.ylim(0.5, 1.01)    
    plt.show()


def plot_sapimouse_counts_boxplot():
    df = pd.read_csv("C:/Anaconda projects/afeCAPTCHA/analysis/statistics/sapimouse_count_1min.csv")
    # df = pd.read_csv("C:/Anaconda projects/afeCAPTCHA/results/sapimouse_count_3min.csv")

    sns.boxplot(data=df, linewidth=2)

    plt.title('OCSVM eredménye SapiMouse adathalmazra', fontsize=30)
    plt.ylabel('AUC érték', fontsize=30)
    plt.xlabel('SapiMouse 1min', fontsize=30)
    plt.xticks(fontsize=28)
    # plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=28)
    # plt.ylim(0.5, 1.01)    
    plt.show()


def plot_occ_aggregated_blocks_result():
    df = pd.read_csv("C:/Anaconda projects/afeCAPTCHA/results/agg_blocks.csv")
    df['block_num'] = df['block_num'].astype(int)

    sns.lineplot(y="value", x="block_num", hue="Szurok szama", data=df)

    plt.title('Aggregált blokkokkal mért eredmény', fontsize=30)
    plt.ylabel('Átlag AUC', fontsize=30)
    plt.xlabel('Aggregált blokkok száma', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=14, loc='lower right')
    plt.show()


def plot_histogramm_blocks_distributions():
    # df = pd.read_csv("C:/Anaconda projects/afeCAPTCHA/analysis/statistics/sapimouse_count_1min.csv")
    df = pd.read_csv("C:/Anaconda projects/afeCAPTCHA/analysis/statistics/sapimouse_count_3min.csv")

    plt.hist(df["blocks_num"], density=False, bins=10, histtype='bar', rwidth=0.95, edgecolor='black', linewidth=1.2)

    plt.title('Hisztogram: SapiMouse 3min', fontsize=30)
    plt.ylabel('Gyakoriság', fontsize=30)
    plt.xlabel('Blokkok száma', fontsize=30)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.legend(fontsize=14, loc='lower right')
    plt.show()



if __name__ == "__main__":

    # train_fcn()
    # train_test_ocsvm()

    nb_filters = 128
    agg_block_num = 10

    train_dset_path = 'c:/Anaconda projects/afeCAPTCHA/datasets/extracted_features/sapimouse_nb_f' + str(nb_filters) + '_3min.csv'
    test_dset_path = 'c:/Anaconda projects/afeCAPTCHA/datasets/extracted_features/sapimouse_nb_f' + str(nb_filters) + '_1min.csv'

    # train_test_ocsvm_with_predefined_features(train_dset_path, test_dset_path)
    # plot_results_boxplot()
    # plot_sapimouse_counts_boxplot()
    plot_occ_aggregated_blocks_result()
    # plot_histogramm_blocks_distributions()