"""
These functions help export the results of the training process of the RL agent
"""

from tensorboard.backend.event_processing import event_accumulator
import os
import pandas as pd

def merge_results(file_name):
    """
    Combines the different files provided by tensorboard and the metrics exported by the model
    :param file_name: str, name of the file (see algorithm, date+name)
    :return: pandas df, combined df
    """
    # Metrics (mt) csv files
    mt_path = 'results/evaluation_metrics_data/'
    mt_file_name = file_name + '.csv'
    mt_file = mt_path + mt_file_name

    # Load the TB data
    # files are large (200+ Mb, might take a while)
    tb_df = extract_tb(file_name, export=False)

    # Load the MT data & convert to pandas
    mt_column_list = ["match", "infeasible_match", "mismatch", "notinstock", "total_removed"]
    mt_df = pd.read_csv(mt_file, names=mt_column_list)

    # Combine the data
    df_combined = pd.concat([tb_df, mt_df], axis=1, sort=False)

    # Save the data
    df_combined_file_name = 'results/combined_data/' + file_name + '.csv'
    df_combined.to_csv(df_combined_file_name, index=False)
    return df_combined


def extract_tb(file_name, export = True) -> pd.DataFrame:
    """
    Extract the information by the tensorboard. Tensorboard stores it in its own files. Use export these files.
    :param file_name: file name of the agent that is trained and stored in tensorboard
    :return: tb_df: pd.DataFrame: dataframe consisiting of the tensborboard data
    """
    # Import the correct files
    # Tensorboard (tb) event file
    tb_path = 'results/tensorboard_data/'
    tb_file_name = file_name + '_1'  # Assuming, the name is changed every time
    for root, dirs, files in os.walk(tb_path):  # Find the file
        if tb_file_name in dirs:
            tb_file_path = os.path.join(root, tb_file_name)
    tb_files = sorted(os.listdir(tb_file_path))

    # Loop through all generated TensorBoard files
    for idx, tb_file in enumerate(tb_files):

        tb_event_file = tb_file_path + '/' + tb_file

        # Load the TB data
        # files are large (200+ Mb, might take a while)
        print(f'start import from tensorboard event {idx}')
        ea = event_accumulator.EventAccumulator(tb_event_file)
        ea.Reload()  # loads events from file
        ea.Tags()
        print(f'import from tensorboard event {idx} done')

        if idx == 0:
            # Convert to pandas
            tb_df = pd.DataFrame(ea.Scalars('episode_reward'))
        else:
            tb_df = pd.concat([tb_df, pd.DataFrame(ea.Scalars('episode_reward'))])
    if export:
        tb_export_file_name = 'results/combined_data/' + file_name + 'not_combined.csv'
        tb_df.to_csv(tb_export_file_name, index=False)
        print('Reward from tensorboard stored in combined data')
    return tb_df
