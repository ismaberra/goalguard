import argparse
import os
import subprocess
import shutil

def run_detection_on_all_videos(weights_path, source_folder, conf_threshold, output_project_path):
    video_files = [f for f in os.listdir(source_folder) if f.endswith('.mp4')]
    for video_file in video_files:
        run_detection(weights_path, video_file, conf_threshold, source_folder, output_project_path)

def run_detection(weights_path, source_video, conf_threshold, input_project_path, output_project_path):
    video_number = source_video.split('-')[2].split('.')[0]  
    output_name = f"{source_video[:-4]}-YOLOv9_{video_number}"  

    source_path = os.path.join(input_project_path, source_video)
    cmd = [
        'python', 'detection.py',
        '--source', source_path,
        '--weights', weights_path,
        '--conf', str(conf_threshold),
        '--project', output_project_path,
        '--name', output_name,
        '--exist-ok'
    ]
    print(f"Running detection on {source_path}...")
    subprocess.run(cmd)

    output_dir = os.path.join(output_project_path, output_name)
    output_files = os.listdir(output_dir)
    for file in output_files:
        if file.endswith('.mp4'):
            original_path = os.path.join(output_dir, file)
            target_path = os.path.join(output_dir, f"{output_name}.mp4")
            shutil.move(original_path, target_path)
            print(f"Renamed output video to {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv9 detection on multiple videos.")
    parser.add_argument('--source-folder', type=str, required=True, help='Folder containing source videos for detection')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, 'yolov9-e.pt')
    video_number = args.source_folder.split('_')[-1] 
    input_project_path = os.path.join(script_dir, f'Videos_RESULT_{video_number}')
    output_project_path = os.path.join(script_dir, f'Detection_RESULT_{video_number}')
    os.makedirs(input_project_path, exist_ok=True)
    os.makedirs(output_project_path, exist_ok=True)

    run_detection_on_all_videos(weights_path, input_project_path, args.conf, output_project_path)
