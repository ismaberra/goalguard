import pandas as pd
import glob
import os

file_pattern = '/Users/tanguydeclety/Desktop/Final_Results/results_*.csv'
output_path = '/Users/tanguydeclety/Desktop/Final_Results/combined_results.csv'

file_paths = sorted(glob.glob(file_pattern))
all_data = []
penalty_count = 1

for file_path in file_paths:
    if os.path.exists(file_path):
        try:
            print(f"Reading file: {file_path}")
            df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip')
            df['Video Name'] = df.apply(lambda row: f'penalty_{penalty_count + row.name}.mp4', axis=1)
            penalty_count += len(df)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    else:
        print(f"File {file_path} does not exist.")

if all_data:
    result_df = pd.concat(all_data, ignore_index=True)
    result_df.to_csv(output_path, index=False)
    print(f"CSV files have been merged and saved as {output_path}")

    total_penalties = len(result_df)
    zone_counts = result_df['Zone'].value_counts()
    zones = ['TR', 'TC', 'TL', 'BR', 'BC', 'BL']
    
    print("Zone percentages:")
    for zone in zones:
        count = zone_counts.get(zone, 0)
        percentage = (count / total_penalties) * 100
        print(f"{zone}: {percentage:.2f}%")
else:
    print("No valid data found to combine.")