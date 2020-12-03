import pandas as pd
import os

c_dataset_path = 'c:/Anaconda projects/afeCAPTCHA/datasets/sapimouse_v3/'
c_stateless_time = 1000
c_min_mouse_event_count = 6
c_block_size = 128


def get_velocities_from_raw_data(username, session_name):
    session_path = c_dataset_path + '/' + username + '/' + session_name
    raw_df = pd.read_csv(session_path)

    raw_df = raw_df.drop_duplicates().reset_index()

    start_pos = 0
    end_pos = 1
    stop_states = ['Pressed', 'Released']
    actual_state_ind = 0
    filtered_df = pd.DataFrame(columns=['vx', 'vy'])

    while True:

        while end_pos < raw_df.shape[0] and \
                raw_df['state'][end_pos] != stop_states[actual_state_ind] and \
                raw_df['client timestamp'][end_pos] - raw_df['client timestamp'][end_pos - 1] <= c_stateless_time:
            end_pos = end_pos + 1

        if end_pos >= raw_df.shape[0]:
            break

        tmp_df = raw_df[['client timestamp', 'x', 'y']][start_pos : end_pos].diff()
        tmp_df['vx'] = tmp_df['x'] / tmp_df['client timestamp']
        tmp_df['vy'] = tmp_df['y'] / tmp_df['client timestamp']
        tmp_df = tmp_df.drop( tmp_df[(tmp_df['vx'] == 0.0) & (tmp_df['vy'] == 0.0)].index )

        if tmp_df.shape[0] > c_min_mouse_event_count:
            filtered_df = pd.concat([filtered_df, tmp_df[['vx', 'vy']][1:]])

        if raw_df['state'][end_pos] == stop_states[actual_state_ind]:
            actual_state_ind = (actual_state_ind + 1) % 2
        
        start_pos = end_pos
        end_pos = end_pos + 1

    return filtered_df.reset_index()


def dump_user_velocities_to_csv(username, session_name, output_filepath):

    velocities_df = get_velocities_from_raw_data(username, session_name)
    print(velocities_df.shape[0])
    
    if not os.path.exists(output_filepath + '/' + username):
        os.makedirs(output_filepath + '/' + username)

    f_out = open(output_filepath + '/' + username + '/' + session_name, 'w')
    for i in range(velocities_df.shape[0] // c_block_size):

        for j in range(c_block_size):
            f_out.write(str(velocities_df.iloc[i * c_block_size + j]['vx']) + ',')

        for j in range(c_block_size):
            f_out.write(str(velocities_df.iloc[i * c_block_size + j]['vy']) + ',')

        f_out.write(username + '\n')

    f_out.close()


def dump_all_user_velocities_to_csv(output_file_path, session_type):

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    f_out = open(output_file_path + '/sapimouse_' + session_type, 'w')
    for username in os.listdir(c_dataset_path):
        print(username)

        for session_name in os.listdir(c_dataset_path + '/' + username):

            if session_name[-8:] == session_type:
                velocities_df = get_velocities_from_raw_data(username, session_name)

                for i in range(velocities_df.shape[0] // c_block_size):

                    for j in range(c_block_size):
                        f_out.write(str(velocities_df.iloc[i * c_block_size + j]['vx']) + ',')

                    for j in range(c_block_size):
                        f_out.write(str(velocities_df.iloc[i * c_block_size + j]['vy']) + ',')

                    f_out.write(str(int(username[-3:])) + '\n')

    f_out.close()




if __name__ == "__main__":
    # df = get_velocities_from_raw_data('user006', 'session_2020_05_25_1min.csv')
    # df.to_csv('out.csv')
    output_files_path = 'C:/Anaconda projects/afeCAPTCHA/datasets/preprocessed_dataset'
    # dump_user_velocities_to_csv('user006', 'session_2020_05_25_1min.csv', output_files_path)
    # dump_all_user_velocities_to_csv(output_files_path, '1min.csv')
    dump_all_user_velocities_to_csv(output_files_path, '3min.csv')