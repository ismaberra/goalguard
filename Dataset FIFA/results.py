import argparse
import csv
import os
import re

def get_shoot_number(folder_name):
    match = re.match(r'result-shoot-(\d+)-YOLOv9_\d+', folder_name)
    return int(match.group(1)) if match else None

def summarize_detection_results(input_dir, output_csv):
    summary_data = []
    sorted_folders = sorted(os.listdir(input_dir))
    for folder_name in sorted_folders:
        shoot_number = get_shoot_number(folder_name)
        if shoot_number is not None:
            folder_path = os.path.join(input_dir, folder_name)
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            for csv_file in csv_files:
                csv_file_path = os.path.join(folder_path, csv_file)
                with open(csv_file_path, mode='r', newline='') as file:
                    csv_reader = list(csv.reader(file))
                    if csv_reader:
                        last_row = csv_reader[-1]
                        if last_row:
                            result_zone = last_row[-1]
                            summary_data.append([shoot_number, result_zone])

    summary_data.sort(key=lambda x: x[0])

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Video Name', 'Zone'])
        for data in summary_data:
            video_name = f"penalty_{data[0]}.mp4"
            writer.writerow([video_name, data[1]])

    print(f"Results CSV file created at {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Summarize detection results into a CSV file.")
    parser.add_argument('--source-folder', type=str, required=True, help='Path to the source folder containing detection results.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(script_dir, args.source_folder)
    video_number = args.source_folder.split('_')[-1]
    output_csv_path = os.path.join(input_directory, f'results_{video_number}.csv')

    summarize_detection_results(input_directory, output_csv_path)

if __name__ == "__main__":
    main()

