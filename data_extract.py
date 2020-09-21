from tensorboard.backend.event_processing import event_accumulator
import os
import pandas as pd

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def merge_results(file_name):
    # Import the correct files
    # Tensorboard (tb) event file
    tb_path = 'results/tensorboard_data/'
    tb_file_name = file_name + '_1'
    for root, dirs, files in os.walk(tb_path):
        if tb_file_name in dirs:
            tb_file_path = os.path.join(root, tb_file_name)
    tb_file = os.listdir(tb_file_path)[0]
    tb_event_file = tb_file_path + '/' + tb_file

    # Metrics (mt) csv files
    mt_path = 'results/evaluation_metrics_data/'
    mt_file_name = file_name + '.csv'
    mt_file = mt_path + mt_file_name

    # Load the TB data
    # files are large (200+ Mb, might take a while)
    print('start import from tensorboard event')
    ea = event_accumulator.EventAccumulator(tb_event_file)
    ea.Reload()  # loads events from file
    ea.Tags()
    print('import from tensorboard event done')
    # Convert to pandas
    tb_df = pd.DataFrame(ea.Scalars('episode_reward'))

    # Load the MT data & convert to pandas
    mt_column_list = ["match", "infeasible_match", "mismatch", "notinstock", "total_removed"]
    mt_df = pd.read_csv(mt_file, names=mt_column_list)

    # Combine the data
    df_combined = pd.concat([tb_df, mt_df], axis=1, sort=False)

    # Save the data
    df_combined_file_name = 'results/combined_data/' + file_name + '.csv'
    df_combined.to_csv(df_combined_file_name, index=False)
    return df_combined


# test_result = merge_results('2020_09_20_13_55_3c_bloodgroup_5m_method2_withoutreset')