import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def plot_data_from_file(file_path):
    # Load the data, assuming the last column is text
    data = pd.read_csv(file_path, header=None)
    rep_size=len(set(data[data.columns[-1]]))
    data.drop(data.columns[-1], axis=1, inplace=True)  # Drop the last column (text)

    # Number of numerical columns
    num_columns = data.shape[1]

    # Create a subplot for each column
    fig, axes = plt.subplots(num_columns, 1, figsize=(10, 6 * num_columns))
    
    # In case there is only one column, axes will not be an array, so we convert it
    if num_columns == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        idx=0
        ax.scatter(np.asarray(data.index,dtype=np.int64)%rep_size, data[i], label=f'Column {i+1}')
        ax.set_title(f'Column {i+1}')
        ax.set_xlabel('ID Number')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.suptitle(f'Data from {os.path.basename(file_path)}')

    # Save the plot to a file
    plt.savefig(file_path.replace('.txt', '.png'))
    plt.close()

def scan_and_plot(directory):
    # Scan for .txt files in the given directory
    txt_files = glob.glob(os.path.join(directory, '*.txt'))

    # Process each file
    for file in txt_files:
        print(f'Processing {file}...')
        plot_data_from_file(file)
        print(f'Plot saved for {file}')
# Replace 'your_folder_path' with the path to the folder containing the .txt files
scan_and_plot('./')