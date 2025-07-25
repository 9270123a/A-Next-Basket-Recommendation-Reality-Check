import pandas as pd
import os

# Define the file paths
file_paths = [
    r"C:\Users\user\Desktop\Funds data\Fund_holding_202201-202212.csv",
    r"C:\Users\user\Desktop\Funds data\Fund_holding_202301-202312.csv",
    r"C:\Users\user\Desktop\Funds data\Fund_holding_202401-202410.csv",
    r"C:\Users\user\Desktop\Funds data\Fund_holding_201501-201512 (1).csv"
]

# Define the output directory
output_dir = r"C:\Users\user\Desktop\Funds data"
output_file = os.path.join(output_dir, "Combined_Fund_holdings.csv")

# Create an empty list to store individual dataframes
dfs = []

# Read each CSV file and append to the list
for file_path in file_paths:
    try:
        print(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        dfs.append(df)
        print(f"Successfully read {len(df)} rows from {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Concatenate all dataframes
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save the combined dataframe to a CSV file
    try:
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully saved combined data to {output_file}")
        print(f"Total rows in combined file: {len(combined_df)}")
    except Exception as e:
        print(f"Error saving combined file: {e}")
else:
    print("No data was read from the files. Please check file paths and formats.")