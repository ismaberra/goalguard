import argparse
import pandas as pd
import cv2
import os

def segment_video_by_frame(source_video, csv_path, cut_output_folder, result_output_folder, frame_rate):
    df = pd.read_csv(csv_path)
    frame_numbers = df['Frame'].tolist()

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  
    
    for i, frame_number in enumerate(frame_numbers, start=1):
        print(f"Processing frame {frame_number} for video segment {i}")
        
        start_frame_timelapse = max(frame_number - 2 * 30, 0)
        end_frame_timelapse = frame_number - 1
        
        start_frame_post = frame_number
        end_frame_post = min(frame_number + int(0.7 * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        timelapse_output_name = os.path.join(cut_output_folder, f"cut-shoot-{i}.mp4")
        post_event_output_name = os.path.join(result_output_folder, f"result-shoot-{i}.mp4")
        
        out_timelapse = cv2.VideoWriter(timelapse_output_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(3)), int(cap.get(4))))
        out_post_event = cv2.VideoWriter(post_event_output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_timelapse)
        for f in range(start_frame_timelapse, end_frame_timelapse + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if (f - start_frame_timelapse) % 2 == 0:
                out_timelapse.write(frame)
        out_timelapse.release()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_post)
        for f in range(start_frame_post, end_frame_post + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out_post_event.write(frame)
        out_post_event.release()
        print(f"Result shoot {i} saved: {post_event_output_name}")

    cap.release()
    print("Finished processing all segments.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create timelapses and post-event videos from detected frames.")
    parser.add_argument('--source', type=str, required=True, help='Source video file name')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_number = args.source.split('-')[1].split('.')[0]
    cut_output_folder = os.path.join(script_dir, f'Videos_CUT_{video_number}')
    result_output_folder = os.path.join(script_dir, f'Videos_RESULT_{video_number}')
    os.makedirs(cut_output_folder, exist_ok=True)
    os.makedirs(result_output_folder, exist_ok=True)
    
    segment_video_by_frame(args.source, os.path.join(script_dir, 'runs', 'detect', f'detected_frames_{video_number}.csv'), cut_output_folder, result_output_folder, frame_rate=60)

