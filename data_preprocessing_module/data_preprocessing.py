import pandas as pd
import numpy as np
import os
import re

from tensorflow.keras.models import load_model

import feature_extractor.fcn as fcn_model
import src.settings as stt_src

import data_preprocessing_module.settings as stt


def get_transformed_data_from_raw_data(username, session_name, max_block_num_to_read):
    session_path = stt.DATASET_PATH + '/' + username + '/' + session_name
    max_row_num = max_block_num_to_read * stt.BLOCK_SIZE

    raw_df = pd.read_csv(session_path, nrows=max_row_num + stt.ROW_NUM_TOL)
    raw_df = raw_df.drop_duplicates().reset_index()

    start_pos = 0
    end_pos = 1
    stop_states = ['Pressed', 'Released']
    actual_state_ind = 0
    filtered_df = pd.DataFrame(columns=['x_calc', 'y_calc'])

    while True:

        while end_pos < raw_df.shape[0] and \
                raw_df['state'][end_pos] != stop_states[actual_state_ind] and \
                raw_df['client timestamp'][end_pos] - raw_df['client timestamp'][end_pos - 1] <= stt.STATELESS_TIME:
            end_pos = end_pos + 1

        if end_pos >= raw_df.shape[0] or filtered_df.shape[0] > max_row_num:
            break

        tmp_df = transform_raw_data_type( raw_df[['client timestamp', 'x', 'y']][start_pos : end_pos] )
        if tmp_df.shape[0] > stt.MIN_MOUSE_EVENT_COUNT:
            filtered_df = pd.concat([ filtered_df, tmp_df ])

        if raw_df['state'][end_pos] == stop_states[actual_state_ind]:
            actual_state_ind = (actual_state_ind + 1) % 2
        
        start_pos = end_pos
        end_pos = end_pos + 1

    return filtered_df[:max_row_num]


def calculate_deviation_from_data(raw_df):
    # we need to drop the first NaN row with the .iloc[1:] command
    tmp_df = raw_df.diff().iloc[1:]
    tmp_df = tmp_df.rename(columns={"x": "x_calc", "y": "y_calc"})
    tmp_df = tmp_df.drop( tmp_df[ ((tmp_df['x_calc'] == 0.0) & (tmp_df['y_calc'] == 0.0) | (tmp_df['client timestamp'] == 0.0)) ].index )

    if stt.SEL_RAW_DATA_TYPE == stt.RawDataType.ABS_DX_DY:
        return tmp_df[['x_calc', 'y_calc']].abs()
    else:
        return tmp_df[['x_calc', 'y_calc']]


def calculate_velocities_from_data(raw_df):
    # we need to drop the first NaN row with the .iloc[1:] command
    tmp_df = raw_df.diff().iloc[1:]
    tmp_df['x_calc'] = tmp_df['x'] / tmp_df['client timestamp']
    tmp_df['y_calc'] = tmp_df['y'] / tmp_df['client timestamp']
    tmp_df = tmp_df.drop( tmp_df[ ((tmp_df['x_calc'] == 0.0) & (tmp_df['y_calc'] == 0.0) | (tmp_df['client timestamp'] == 0.0)) ].index )

    if stt.SEL_RAW_DATA_TYPE == stt.RawDataType.ABS_VX_VY:
        return tmp_df[['x_calc', 'y_calc']].abs()
    else:
        return tmp_df[['x_calc', 'y_calc']]


def transform_raw_data_type(raw_data):
    switcher = {
            0: calculate_deviation_from_data,
            1: calculate_deviation_from_data,
            2: calculate_velocities_from_data,
            3: calculate_velocities_from_data
        }

    data_transformator = switcher.get(stt.SEL_RAW_DATA_TYPE.value, lambda: 'Not a valid raw data type!')
    return data_transformator(raw_data)


def dump_all_user_transformed_data_to_csv(output_file_path, session_type):

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    dataset_name = stt.DATASET_PATH.split('/')[-1]
    f_out = open(output_file_path + '/' + dataset_name + '_' + str(stt.SEL_RAW_DATA_TYPE.name) + '_' + stt.SEL_SCALER.name + '_' + session_type, 'w')

    for username in os.listdir(stt.DATASET_PATH):
        print(username)

        actual_block_num_dumped = 0
        for session_name in os.listdir(stt.DATASET_PATH + '/' + username):

            if session_name[-8:] == session_type or session_type == 'all.csv':
                transformed_arr = scale_dataset( reshape_df_to_np( get_transformed_data_from_raw_data(username, session_name, stt.MAX_BLOCK_NUM - actual_block_num_dumped) ) )
                actual_block_num_dumped = actual_block_num_dumped + transformed_arr.shape[0]

                for i in range(transformed_arr.shape[0]):

                    for j in range(stt.BLOCK_SIZE):
                        f_out.write(str(transformed_arr[i, j, 0]) + ',')

                    for j in range(stt.BLOCK_SIZE):
                        f_out.write(str(transformed_arr[i, j, 1]) + ',')

                    f_out.write(str(get_id_from_str(username)) + '\n')

            if actual_block_num_dumped >= stt.MAX_BLOCK_NUM:
                break

    f_out.close()


def reshape_df_to_np(transformed_df):
    transformed_arr = transformed_df.to_numpy()
    max_movement_count = transformed_arr.shape[0] // stt.BLOCK_SIZE

    return np.reshape(transformed_arr[:max_movement_count * stt.BLOCK_SIZE], (max_movement_count, stt.BLOCK_SIZE, 2))


def scale_dataset(data):

    switcher = {
            0: no_scaler,
            1: standard_scaler,
        }

    scaler = switcher.get(stt.SEL_SCALER.value, lambda: 'Not a valid raw data type!')

    for i in range(data.shape[0]):
        data[i] = scaler(data[i])

    return data


def no_scaler(data):
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


def rename_files (folder_path):
    print('Renaming folders to proper form...')
    for sub_folder_name in os.listdir(folder_path):
        # user_id = int(re.search(r'\d+', sub_folder_name).group())
        user_id = get_id_from_str(sub_folder_name)
        new_folder_name = 'user'
        if user_id < 10:
            new_folder_name += '00' + str(user_id)
        elif user_id < 100:
            new_folder_name += '0' + str(user_id)
        else:
            new_folder_name += str(user_id)
        os.rename (folder_path + '/' + sub_folder_name, folder_path + '/' + new_folder_name)
    
    print('Renaming folders done.')


def get_id_from_str(raw_str):
    return int(re.search(r'\d+', raw_str).group())


def extract_features_from_raw_data():
    print("Extracting features...")
    model_path = stt_src.OUTPUT_MODEL_DIRECTORY + '/' + fcn_model.stt_fcn.BEST_MODEL_NAME + str(fcn_model.stt_fcn.NB_FILTERS) + '.hdf5'
    feature_extractor = get_feature_extractor(model_path)

    df_raw_data = pd.read_csv(stt.INPUT_EXTRACT_FEATURES_PATH, header=None)
    
    output_file_path = stt.OUTPUT_EXTRACTED_FEATURES_PATH + '/extracted_features_f' + str(fcn_model.stt_fcn.NB_FILTERS) + '_' + stt.INPUT_EXTRACT_FEATURES_PATH[-8:]
    f_out = open(output_file_path, 'w')
    for user_id in range(1, stt.NB_CLASSES + 1):
        train_raw_data = get_raw_data_by_user_id(df_raw_data, user_id)
        train_raw_data = train_raw_data.iloc[:, 0:(fcn_model.stt_fcn.NB_FILTERS*2)].to_numpy().reshape((train_raw_data.shape[0], fcn_model.stt_fcn.NB_FILTERS, 2), order='F')

        tmp_features = get_features_from_model(feature_extractor, train_raw_data)
        
        for i in range(tmp_features.shape[0]):
            for j in range(tmp_features.shape[1]):
                f_out.write(str(tmp_features[i, j]) + ',')
            f_out.write(str(user_id) + '\n')

    f_out.close()
    print("Features extracted successfully.")


def get_raw_data_by_user_id(df, user_id):
    user_id_index = fcn_model.stt_fcn.NB_FILTERS * 2
    data_X = df.loc[df.iloc[:, user_id_index] == user_id]
    return data_X.iloc[:, :user_id_index]


def get_feature_extractor(model_path):

    print('Loaded model path:', model_path)

    model = load_model(model_path)
    model._layers.pop()
    model.outputs = [model.layers[-1].output]

    return model


def get_features_from_model(model, raw_data):
    return model.predict(raw_data)



if __name__ == "__main__":
    # rename_files (stt.DATASET_PATH)
    # dump_all_user_transformed_data_to_csv(stt.PREPROCESSED_DATASET_PATH, '1min.csv')
    # dump_all_user_transformed_data_to_csv(stt.PREPROCESSED_DATASET_PATH, '3min.csv')
    # dump_all_user_transformed_data_to_csv(stt.PREPROCESSED_DATASET_PATH, 'all.csv')
    extract_features_from_raw_data()