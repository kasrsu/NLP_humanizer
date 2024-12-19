import pandas as pd
import os

def merge_csv_files(directory_path, output_file):
    # List to hold dataframes
    dataframes = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    
    # Concatenate all dataframes row-wise
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)

# Example usage
directory_path = '../data_extraction/content extraction'
output_file = 'final_output.csv'
merge_csv_files(directory_path, output_file)