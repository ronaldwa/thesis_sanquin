import matplotlib.pyplot as plt
import numpy as np
from sanquin_inventory import find_compatible_blood_list
import pandas as pd
from datetime import datetime

time_string = datetime.now().strftime("%Y_%m_%d_%H_%M")


# todo comments and information

def create_match_matrix(match_dict, blood_keys):
    match_matrix = []
    match_matrix_percentage = []
    for blood_group1 in blood_keys:
        blood_group_list = []
        for blood_group2 in blood_keys:
            if blood_group1 in match_dict:
                if blood_group2 in match_dict[blood_group1]:
                    blood_group_list.append(match_dict[blood_group1][blood_group2])
                else:
                    blood_group_list.append(0)
            else:
                blood_group_list = [0] * len(blood_keys)

        blood_group_list_percentage = [round(float(i) / sum(blood_group_list), 2) for i in blood_group_list]

        match_matrix.append(blood_group_list)
        match_matrix_percentage.append(blood_group_list_percentage)
    return match_matrix, match_matrix_percentage


# create_match_matrix(result['match'], list(demand_distribution[0].keys()))


def answered_insight(blood_binary, match_result_dict):
    match_dict = match_result_dict[blood_binary]

    values = match_dict.values()
    keys = match_dict.keys()

    zipped_lists = zip(values, keys)
    sorted_pairs = sorted(zipped_lists, reverse=True)

    tuples = zip(*sorted_pairs)
    values, keys = [list(tuple) for tuple in tuples]

    y_pos = np.arange(len(keys))

    # Create bars
    plt.bar(y_pos, values)

    # Create names on the x-axis
    plt.xticks(y_pos, keys)

    plt.xlabel('Matched blood groups')
    plt.ylabel("Number of matches")
    plt.title(f"Matched blood for {blood_binary}")

    # Show graphic
    plt.show()


def match_matrix(blood_group_keys, match_result_dict, file_name):
    plt.figure(num=None, figsize=(20, 6), dpi=300, facecolor='w', edgecolor='k')
    # sphinx_gallery_thumbnail_number = 2

    blood_key = blood_group_keys
    data = create_match_matrix(match_result_dict, blood_group_keys)
    match_matrix = np.array(data[0])
    match_matrix_percentage = np.array(data[1])

    fig, ax = plt.subplots()
    im = ax.imshow(match_matrix_percentage, cmap='Reds')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(blood_key)))
    ax.set_yticks(np.arange(len(blood_key)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(blood_key)
    ax.set_yticklabels(blood_key)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(blood_key)):
        _, feasible_action = find_compatible_blood_list(blood_group_keys, i)

        for j in range(len(blood_key)):
            if j in feasible_action:
                if match_matrix_percentage[i, j] < 0.5:
                    text = ax.text(j, i,
                                   str(round(match_matrix_percentage[i, j] * 100)) + '% \n' + str(match_matrix[i, j]),
                                   ha="center", va="center", color="black")
                else:
                    text = ax.text(j, i,
                                   str(round(match_matrix_percentage[i, j] * 100)) + '% \n' + str(match_matrix[i, j]),
                                   ha="center", va="center", color="white")
            else:
                text = ax.text(j, i, '-',
                               ha="center", va="center", color="black")

    ax.set_title("matched blood in absolute numbers")

    # Set common labels
    ax.set_xlabel('Provided blood group')
    ax.set_ylabel('Requested blood group')

    fig.tight_layout()
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.savefig('results/' + time_string + '_' + file_name + '_match_matrix.png')
    plt.show()


def age_histogram(age_dict, blood_binary):
    age_dict = age_dict[blood_binary]
    mean_age = 0
    counter = 0
    for key, value in age_dict.items():
        mean_age += key * value
        counter += value
    mean_age = round(mean_age / counter, 1)
    plt.bar(list(age_dict.keys()), age_dict.values())
    plt.ylabel("Number RBCs")
    plt.xlabel("Age of RBC")

    plt.title(f"Age histogram for {blood_binary}, average age={mean_age}, n={counter}")
    plt.show()


def age_subplot(age_dict, file_name):
    plt.figure(num=None, figsize=(20, 6), dpi=300, facecolor='w', edgecolor='k')

    number_per_row = int(len(age_dict.keys()) / 2)
    fig, axs = plt.subplots(2, number_per_row)

    total_counter = 0
    total_mean_age = 0

    for x, key in enumerate(age_dict):
        if x < number_per_row:
            y = 0
        else:
            x = x - number_per_row
            y = 1

        axs[y, x].bar(list(age_dict[key].keys()), age_dict[key].values())
        mean_age = 0
        counter = 0
        for key2, value in age_dict[key].items():
            mean_age += key2 * value
            counter += value
            total_counter += value
            total_mean_age += key2 * value
        mean_age = round(mean_age / counter, 1)
        axs[y, x].set_title(f"{key} \n average={mean_age}, n={counter}")

    for ax in axs.flat:
        ax.set(xlabel='RBC age', ylabel='number of RBCs')

    total_mean_age = round(total_mean_age / total_counter, 1)
    fig.suptitle(f"Age histogram for all blood groups \n average={total_mean_age}, n={total_counter}")

    plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.savefig('results/' + time_string + '_' + file_name + '_age_subplot.png')
    plt.show()


def flow_trough_subplot(eval_metrics, file_name):
    plt.figure(num=None, figsize=(20, 6), dpi=300, facecolor='w', edgecolor='k')

    number_per_row = int(len(eval_metrics['donated'].keys()) / 2)
    fig, axs = plt.subplots(2, number_per_row)

    for x, blood_binary in enumerate(eval_metrics['age']):
        if x < number_per_row:
            y = 0
        else:
            x = x - number_per_row
            y = 1

        donated = eval_metrics['donated'][blood_binary]
        requested = eval_metrics['requested'][blood_binary]
        provided = eval_metrics['provided'][blood_binary]
        if blood_binary in eval_metrics['match'][blood_binary]:
            exact_match = eval_metrics['match'][blood_binary][blood_binary]
        else:
            exact_match = 0
        removed = eval_metrics['removed'][blood_binary]
        infeasible = eval_metrics['infeasible'][blood_binary]

        values = [donated, requested, provided, exact_match, removed, infeasible]
        keys = ['donated', 'requested', 'provided', 'exact_match', 'removed', 'infeasible']

        axs[y, x].bar(keys, values)
        axs[y, x].set_title(blood_binary)

    for ax in axs.flat:
        ax.set(ylabel='number of RBCs')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.suptitle(f"Flowthrough metrics of blood")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.savefig('results/' + time_string + '_' + file_name + '_flow_trough.png')
    plt.show()


def age_table(result):
    columns = ['blood_group'] + list(range(1, 36))
    df_age = pd.DataFrame(columns=columns)

    for blood_binary, value in result['age'].items():
        row = [blood_binary]
        for i in list(range(1, 36)):
            if i in result['age'][blood_binary]:
                row.append(result['age'][blood_binary][i])
            else:
                row.append(0)
        df_age.loc[len(df_age)] = row
    df_age.loc["Total"] = df_age.sum()
    df_age.at['Total', 'blood_group'] = 'Total'
    df_age = df_age.set_index('blood_group')
    return df_age


# age_table(result)

def flow_metrics(eval_metrics):
    columns = ['blood_group', 'donated', 'requested', 'provided', 'exact match', 'removed', 'infeasible']
    df_flow = pd.DataFrame(columns=columns)

    for blood_binary in eval_metrics['age']:
        donated = eval_metrics['donated'][blood_binary]
        requested = eval_metrics['requested'][blood_binary]
        provided = eval_metrics['provided'][blood_binary]
        if blood_binary in eval_metrics['match'][blood_binary]:
            exact_match = eval_metrics['match'][blood_binary][blood_binary]
        else:
            exact_match = 0
        removed = eval_metrics['removed'][blood_binary]
        infeasible = eval_metrics['infeasible'][blood_binary]

        row = [blood_binary, donated, requested, provided, exact_match, removed, infeasible]
        df_flow.loc[len(df_flow)] = row
    df_flow.loc["Total"] = df_flow.sum()
    df_flow.at['Total', 'blood_group'] = 'Total'
    df_flow = df_flow.set_index('blood_group')

    df_flow['requested'] = df_flow['requested'].astype(int)
    df_flow['removed'] = df_flow['removed'].astype(int)
    return df_flow


# flow_metrics(result)

def match_matrix_table(data, blood_group_keys):
    blood_groups = blood_group_keys
    matrix_absolute = create_match_matrix(data['match'], blood_groups)[0]
    matrix_percentage = create_match_matrix(data['match'], blood_groups)[1]

    df_abs = pd.DataFrame(columns=['blood_group'] + blood_groups)
    df_per = pd.DataFrame(columns=['blood_group'] + blood_groups)

    for x, blood_binary in enumerate(blood_groups):
        row_abs = [blood_binary] + matrix_absolute[x]
        row_per = [blood_binary] + matrix_percentage[x]
        df_abs.loc[len(df_abs)] = row_abs
        df_per.loc[len(df_per)] = row_per
    df_abs = df_abs.set_index('blood_group')
    df_per = df_per.set_index('blood_group')
    return df_abs, df_per


def export_results(result, blood_group_keys, file_name):
    match_matrix(blood_group_keys, result['match'], file_name)
    age_subplot(result['age'], file_name)
    flow_trough_subplot(result, file_name)

    with pd.ExcelWriter('results/' + time_string + '_' + file_name + '.xlsx') as writer:
        flow_metrics(result).to_excel(writer, sheet_name='Flow_metrics')
        age_table(result).to_excel(writer, sheet_name='Issued_age')
        match_matrix_table(result, blood_group_keys)[0].to_excel(writer, sheet_name='match_matrix_absolute')
        match_matrix_table(result, blood_group_keys)[1].to_excel(writer, sheet_name='match_matrix_percentage')
    print(f'results exported into: {file_name}.xlsx')


def smooth(scalars, weight):
    """
    Smoothes the datapoints
    :param scalars: list with datapoints
    :param weight: float between 0 and 1, how smooth
    :return: the smoothed data points
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed


def plot_learning_eval(data_list, labels, file_name, smoothing=0.5, ylim_cutoff=0):
    """
    Plots the data provided in a line chart. Up to 5 different datasets.
    Also includes a smoothing option.
    :param data_list: nested list, data to be shown. [[data1.1, data1.2],[data2]]
    :param labels: list of strings, ['name of data1', 'name of data2']
    :param smoothing: float, number to provide smoothing
    :param ylim_cutoff: cutoff value in limiting. 0.01 is already a lot.
    :return:
    """
    # Initialize lists for scaling the graph
    ylim_list_max = []
    ylim_list_min = []

    # Set the size of the picture
    plt.figure(num=None, figsize=(20, 6), dpi=300, facecolor='w', edgecolor='k')

    # Initialize the colors of the lines in the graph
    color_list = ['blue', 'red', 'green', 'orange', 'black']

    # Loop through the datasets to create a line and possibly a smoothing line
    for idx, data in enumerate(data_list):
        x = list(range(len(data)))
        if smoothing == 0:
            plt.plot(x, data, color=color_list[idx])
        else:
            data_smooth = smooth(data, smoothing)
            plt.plot(x, data_smooth, label=labels[idx], color=color_list[idx])
            plt.plot(x, data, color=color_list[idx], alpha=0.2)

        # Add values to scale the graph
        cutoff_max = round(len(data) * (1 - ylim_cutoff)) - 1
        ylim_list_max.append(sorted(data)[cutoff_max])
        cutoff_min = round(len(data) * (1 - ylim_cutoff)) - 1
        ylim_list_min.append(sorted(data, reverse=True)[cutoff_min])

    plt.legend()

    # Scale the graph
    ylim_value_max = max(ylim_list_max)
    ylim_value_min = min(ylim_list_min)
    min_value = ylim_value_min - ((ylim_value_max - ylim_value_min) / 10)
    max_value = ylim_value_max + ((ylim_value_max - ylim_value_min) / 10)
    if ylim_value_min < 0:
        plt.ylim(min_value, max_value)
    elif ylim_value_min > 20:
        min_value = ylim_value_min - ((ylim_value_max - ylim_value_min) / 10)
        plt.ylim(min_value, max_value)
    else:
        plt.ylim(0, max_value)

    # plt.ylim(ylim_value_min, ylim_value_max)
    plt.xlabel("Episodes")
    plt.savefig('/results/fig/' + file_name + '_reward_function.png')
    plt.show()
