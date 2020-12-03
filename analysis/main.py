import numpy as np 
import pandas as pd

import os


train_dataset_path = 'c:/Anaconda projects/Software_mod/analysis/raw_data/sapimouse_128_3min.csv'
test_dataset_path = 'c:/Anaconda projects/Software_mod/analysis/raw_data/sapimouse_128_1min.csv'

output_folder = 'C:/Anaconda projects/afeCAPTCHA/analysis'
nb_users = 123

def print_user_blocks_num(dataset_path, output_filename):
    print("Calculating user blocks...")
    df = pd.read_csv(dataset_path, header=None)

    results = {}
    for user_id in range(1, 124):
        username = 'user' + str(user_id)
        results[username] = df.loc[df.iloc[:, 256] == user_id].shape[0]

    file = open(output_folder + '/' + output_filename, 'w')
    file.write('username,blocks_num\n')

    for username, nb_blocks in results.items():
        file.write(str(username) + ',' + str(nb_blocks) + '\n')

    file.close()


def print_mouse_movements(dataset_path, output_filename):

    results = {}
    for username in os.listdir(dataset_path):
        df = pd.read_csv(dataset_path + '/' + username)
        results[username[:-4]] = df['movement_length'].mean()

    
    file = open(output_folder + '/' + output_filename, 'w')
    file.write('username,movement_len_avg\n')

    for username, nb_blocks in results.items():
        file.write(str(username) + ',' + str(nb_blocks) + '\n')

    file.close()


if __name__ == "__main__":
    # print_user_blocks_num(test_dataset_path, "sapimouse_1min_blocks_num.csv")
    # print_user_blocks_num(train_dataset_path, "sapimouse_3min_blocks_num.csv")
    print_mouse_movements('C:/Anaconda projects/afeCAPTCHA/analysis/user_mouse_movements', 'movements_len_avg.csv')