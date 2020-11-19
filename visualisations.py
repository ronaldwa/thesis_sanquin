"""
This file contains all functions needed for visualisations and exports to excel.
"""
import matplotlib.pyplot as plt
import numpy as np
from sanquin_inventory import find_compatible_blood_list
import pandas as pd
from datetime import datetime
from math import ceil
from typing import Tuple


def create_match_matrix(match_dict, blood_keys) -> Tuple[list, list]:
    """
    Combine the match matrix based on the information on the matches
    :param match_dict: dict, containing the number of supplied items per blood group for every requested blood group
    :param blood_keys: list, containing the blood groups
    :return: match_matrix, match_matrix_percentage: both lists containing the total absolute numbers and percentages
    """
    # Initialize matrices
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
        if sum(blood_group_list) == 0:
            blood_group_list_percentage = [0.00 for _ in blood_group_list]
        else:
            blood_group_list_percentage = [round(float(i) / sum(blood_group_list), 2) for i in blood_group_list]

        # Add per row to the data matrices
        match_matrix.append(blood_group_list)
        match_matrix_percentage.append(blood_group_list_percentage)
    return match_matrix, match_matrix_percentage


def match_matrix(blood_group_keys, match_result_dict, file_name, show_absolute=True, title=True) -> None:
    """
    Plot the matrix data in a heatmap matrix. Fig is plotted and saved.
    :param blood_group_keys: list, consisting of blood groups in a list
    :param match_result_dict: dict with information from evaluation
    :param file_name: str: name to store the file in
    :param show_absolute: boolean: store the
    :param title:
    """
    # Initialize figure
    plt.figure(num=None, figsize=(9, 9), dpi=300, facecolor='w', edgecolor='k')

    # Get right data
    blood_key = blood_group_keys
    data = create_match_matrix(match_result_dict, blood_group_keys)
    match_matrix = np.array(data[0])
    match_matrix_percentage = np.array(data[1])

    # Initialize plots
    fig, ax = plt.subplots()
    im = ax.imshow(match_matrix_percentage, cmap='Reds')

    # Show ticks
    ax.set_xticks(np.arange(len(blood_key)))
    ax.set_yticks(np.arange(len(blood_key)))

    # Set ticks to right label
    ax.set_xticklabels(blood_key)
    ax.set_yticklabels(blood_key)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data and create annotations
    for i in range(len(blood_key)):
        _, feasible_action = find_compatible_blood_list(blood_group_keys, i)

        for j in range(len(blood_key)):
            if j in feasible_action:
                # Show absolute numbers & percentages
                if show_absolute:
                    # For values <0.05 the text color needs to be black (light background)
                    if match_matrix_percentage[i, j] < 0.5:
                        text = ax.text(j, i,
                                       str(round(match_matrix_percentage[i, j] * 100)) + '% \n' + str(
                                           match_matrix[i, j]),
                                       ha="center", va="center", color="black")
                    # For values >0.05 the text color needs to be white (dark background)
                    else:
                        text = ax.text(j, i,
                                       str(round(match_matrix_percentage[i, j] * 100)) + '% \n' + str(
                                           match_matrix[i, j]),
                                       ha="center", va="center", color="white")
                    if title:
                        ax.set_title("Matched blood in absolute numbers")
                # Show only percentages
                else:
                    # For values <0.05 the text color needs to be black (light background)
                    if match_matrix_percentage[i, j] < 0.5:
                        text = ax.text(j, i,
                                       str(round(match_matrix_percentage[i, j] * 100)) + '%',
                                       ha="center", va="center", color="black")
                    # For values >0.05 the text color needs to be white (dark background)
                    else:
                        text = ax.text(j, i,
                                       str(round(match_matrix_percentage[i, j] * 100)) + '%',
                                       ha="center", va="center", color="white")
                    if title:
                        ax.set_title("Matched blood in percentages")
            else:
                text = ax.text(j, i, '-',
                               ha="center", va="center", color="black")

    # If title is different from default, use the provided string
    if type(title) == str:
        ax.set_title(title)

    # Set common labels
    ax.set_xlabel('Assigned blood group')
    ax.set_ylabel('Requested blood group')

    fig.set_size_inches(6, 6, forward=True)
    fig.tight_layout()

    # Use time string to track all the files
    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    fig.savefig('results/fig/' + time_string + '_' + file_name + '_match_matrix.png')


def age_subplot(age_dict, file_name) -> None:
    """
    Creates and saves a figure of the age histograms of blood per blood group
    :param age_dict: dict: containing blood group per age the number of issued blood: {blood_group : {1: number, 2:...}}
    :param file_name: str: name of the file to export to
    """

    # Initialize figure
    plt.figure(num=None, figsize=(20, 6), dpi=300, facecolor='w', edgecolor='k')

    # Four graphs per row
    columns = 4
    rows = ceil(len(age_dict.keys()) / columns)

    fig, axs = plt.subplots(rows, columns)

    # Initialize counters for calculations in title
    total_counter = 0
    total_mean_age = 0

    # Initialize counters for plot position
    counter_y = 0
    counter_x = 0

    # Loop through all the blood groups and plot
    for key in age_dict:
        if counter_y % columns == 0:
            y = int(ceil(counter_y / columns))
            counter_x = -1
        counter_y += 1
        counter_x += 1

        # Plot the histogram
        axs[y, counter_x].bar(list(age_dict[key].keys()), age_dict[key].values())
        mean_age = 0
        counter = 0

        # Calculate the average age and total number of RBCs used
        for key2, value in age_dict[key].items():
            mean_age += key2 * value
            counter += value
            total_counter += value
            total_mean_age += key2 * value
        if counter != 0:
            mean_age = round(mean_age / counter, 1)
        else:
            mean_age = 0
        # Set title
        axs[y, counter_x].set_title(f"{key} \n average={mean_age}, n={counter}")

    # Add x label to only last row
    for ax in axs.flat:
        ax.set(xlabel='RBC age')

    # Add y label only to first column
    axs[0, 0].set(ylabel='number of RBCs')
    axs[1, 0].set(ylabel='number of RBCs')

    # Create a title for whole subplot, including the total number of RBCs and mean issued age
    total_mean_age = round(total_mean_age / total_counter, 1)
    fig.suptitle(f"Age histogram for all blood groups \n average={total_mean_age}, n={total_counter}")

    # Adjust some sizes
    plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches(18.5, 10.5, forward=True)

    # Use time string to track all the files
    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig('results/fig/' + time_string + '_' + file_name + '_age_subplot.png')


def flow_trough_subplot(eval_metrics, file_name) -> None:
    """
    The flow though plots per bloodgrou a variety of information:
    - The number of requested items
    - The number of supplied items
    - The number of identical matches
    - The number of items perished (outdating)
    - The number of requests that could not be answered (shortage)
    :param eval_metrics: dict: containing all evaluation data
    :param file_name: str: name to use for saving the image
    """
    # Initialize figure
    plt.figure(num=None, figsize=(20, 6), dpi=300, facecolor='w', edgecolor='k')

    # Determine rows and columns, want two rows
    number_per_row = int(len(eval_metrics['donated'].keys()) / 2)
    fig, axs = plt.subplots(2, number_per_row)

    # Loop through the data and plot subplots
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

        # Data and labels for subplot
        values = [donated, requested, provided, exact_match, removed, infeasible]
        keys = ['donated', 'requested', 'provided', 'exact_match', 'removed', 'infeasible']

        # Plot data
        axs[y, x].bar(keys, values)
        axs[y, x].set_title(blood_binary)

    # Set labels for all subplots
    for ax in axs.flat:
        ax.set(ylabel='number of RBCs')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.suptitle(f"Flow through metrics of blood")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches(18.5, 10.5, forward=True)

    # Use time string to track all the files
    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig('results/fig/' + time_string + '_' + file_name + '_flow_trough.png')


def age_table(eval_metrics) -> pd.DataFrame:
    """
    Convert the information in the eval_metrics dictionary to a DataFrame to be stored in csv or excel
    :param eval_metrics: dict: containing all evaluation data
    :return: df_age: pd.DataFrame: consisting of the blood groups as columns and age as rows, the values represent
    the issued items per blood group and age
    """
    # Initialize DataFrame
    columns = ['blood_group'] + list(range(1, 36))
    df_age = pd.DataFrame(columns=columns)

    # Add the ages per blood group to the DataFrame
    for blood_binary, value in eval_metrics['age'].items():
        row = [blood_binary]
        for i in list(range(1, 36)):
            if i in eval_metrics['age'][blood_binary]:
                row.append(eval_metrics['age'][blood_binary][i])
            else:
                row.append(0)
        df_age.loc[len(df_age)] = row

    # Add a total columns
    df_age.loc["Total"] = df_age.sum()
    df_age.at['Total', 'blood_group'] = 'Total'

    df_age = df_age.set_index('blood_group')
    return df_age


def flow_metrics(eval_metrics) -> pd.DataFrame:
    """
    Flow metrics stores a variety of information into a DataFrame:
    - The number of requested items
    - The number of supplied items
    - The number of identical matches
    - The number of items perished (outdating)
    - The number of requests that could not be answered (shortage)
    :param eval_metrics: dict: containing all evaluation data
    :return: df_flow: pd.DataFrame: columns are the different metrics, rows are the blood groups
    """
    # Initialize DataFrame
    columns = ['blood_group', 'donated', 'requested', 'provided', 'exact match', 'removed', 'infeasible']
    df_flow = pd.DataFrame(columns=columns)

    # Add information per blood group as row
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

        # Add all retrieved data as row (blood group)
        row = [blood_binary, donated, requested, provided, exact_match, removed, infeasible]
        df_flow.loc[len(df_flow)] = row

    # Add a total row
    df_flow.loc["Total"] = df_flow.sum()
    df_flow.at['Total', 'blood_group'] = 'Total'
    df_flow = df_flow.set_index('blood_group')

    # Set the right types
    df_flow['requested'] = df_flow['requested'].astype(int)
    df_flow['removed'] = df_flow['removed'].astype(int)
    return df_flow


def match_matrix_table(eval_metrics, blood_group_keys) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses the information on matches to create match matrices. And export this as a DataFrame to export in excel or csv
    :param eval_metrics: dict: containing all evaluation data
    :param blood_group_keys: list: containing the blood group names
    :return: df_abs, df_per: two DataFrames, the first containing the absolute match numbers, the second the percentages
    """
    # Retrieve matrices
    matrix_absolute = create_match_matrix(eval_metrics['match'], blood_group_keys)[0]
    matrix_percentage = create_match_matrix(eval_metrics['match'], blood_group_keys)[1]

    # Initialize DataFrames
    df_abs = pd.DataFrame(columns=['blood_group'] + blood_group_keys)
    df_per = pd.DataFrame(columns=['blood_group'] + blood_group_keys)

    # Add the information per row
    for x, blood_binary in enumerate(blood_group_keys):
        row_abs = [blood_binary] + matrix_absolute[x]
        row_per = [blood_binary] + matrix_percentage[x]
        df_abs.loc[len(df_abs)] = row_abs
        df_per.loc[len(df_per)] = row_per
    df_abs = df_abs.set_index('blood_group')
    df_per = df_per.set_index('blood_group')
    return df_abs, df_per


def export_results(eval_metrics, blood_group_keys, file_name, include_visuals=True) -> None:
    """

    :param eval_metrics: dict: containing all evaluation data
    :param blood_group_keys: list: containing the blood group names
    :param file_name: str: name of the file to store
    :param include_visuals: boolean: True if figures need to be plotted. False if only excel file is necessary
    :return:
    """
    # Use time string to track all the files
    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M")

    if include_visuals:
        match_matrix(blood_group_keys, eval_metrics['match'], file_name)
        age_subplot(eval_metrics['age'], file_name)
        flow_trough_subplot(eval_metrics, file_name)

    with pd.ExcelWriter('results/fig_data_excel/higher_demand/' + time_string + '_' + file_name + '.xlsx') as writer:
        flow_metrics(eval_metrics).to_excel(writer, sheet_name='Flow_metrics')
        age_table(eval_metrics).to_excel(writer, sheet_name='Issued_age')
        match_matrix_table(eval_metrics, blood_group_keys)[0].to_excel(writer, sheet_name='match_matrix_absolute')
        match_matrix_table(eval_metrics, blood_group_keys)[1].to_excel(writer, sheet_name='match_matrix_percentage')
    print(f'results exported into: results/fig_data_excel/higher_demand/' + time_string + '_' + file_name + '.xlsx')


def smooth(scalars, weight) -> list:
    """
    Smoothes the datapoints
    :param scalars: list with datapoints
    :param weight: float between 0 and 1, how smooth
    :return: list of the smoothed data points
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed


def plot_learning_eval(data_list, labels, file_name, smoothing=0.5, ylim_cutoff=0) -> None:
    """
    Plots the data provided in a line chart. Up to 5 different datasets.
    Also includes a smoothing option.
    :param file_name: str: name of the file to store
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

    # Loop through the data sets to create a line and possibly a smoothing line
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

    # plt.legend()

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

    plt.xlabel("Episodes")

    # Use time string to track all the files
    time_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    plt.savefig('results/fig/' + time_string + '_' + file_name + '_reward_function.png')
