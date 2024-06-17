import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import pandas as pd
import cv2
from sttrans import ST_Trans

LABELS = ['TR', 'TL', 'BR', 'BL']

def preprocess_input(penalty_folder):
    frames = []
    for i in range(30):
        frame_file = os.path.join(penalty_folder, f'keypoints_frame_{i}.csv')
        if os.path.exists(frame_file):
            frame_data = np.genfromtxt(frame_file, delimiter=',', skip_header=1)
            if frame_data.ndim == 1:
                frame_data = np.expand_dims(frame_data, axis=0)
            frame_data = frame_data[:, 1:].reshape(-1)
            for j in range(0, frame_data.shape[0], 3):
                x = frame_data[j]
                y = frame_data[j+1]
                viability = frame_data[j+2]
                if x == 0 and y == 0:
                    frame_data[j] = -1
                    frame_data[j+1] = -1
                    frame_data[j+2] = 0
                else:
                    frame_data[j] = x / 1920 if x != 0 else -1
                    frame_data[j+1] = y / 1080 if y != 0 else -1
            frames.append(frame_data)
    frames = np.array(frames)
    return torch.tensor(frames, dtype=torch.float32).reshape(1, 30, -1)

def load_model(model_path):
    input_dim = 36
    num_classes = 4
    num_layers = 4
    nhead = 4
    dim_feedforward = 768
    dropout = 0.38415372018572036
    
    model = ST_Trans(input_dim=input_dim, num_classes=num_classes, num_layers=num_layers, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_true_label(penalty_folder, combined_results_csv):
    combined_results = pd.read_csv(combined_results_csv)
    combined_results['Folder'] = combined_results['Video Name'].apply(lambda x: x.replace('.mp4', ''))
    folder_name = os.path.basename(penalty_folder)
    true_label = combined_results.loc[combined_results['Folder'] == folder_name, 'Zone'].values
    return true_label[0]

def draw_prediction(video_path, predicted_label, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    horizontal_line = 360  
    left_part = 785       
    right_part = 1160

    if predicted_label == 'TL':
        x_center = left_part - 150
        y_center = horizontal_line - 100
    elif predicted_label == 'TR':
        x_center = right_part - 150
        y_center = horizontal_line - 100
    elif predicted_label == 'BL':
        x_center = left_part + 150
        y_center = horizontal_line + 100
    elif predicted_label == 'BR':
        x_center = right_part + 150
        y_center = horizontal_line + 100

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.line(frame, (x_center - 35, y_center - 35), (x_center + 35, y_center + 35), (0, 255, 0), 15)
        frame = cv2.line(frame, (x_center + 35, y_center - 35), (x_center - 35, y_center + 35), (0, 255, 0), 15)

        frame = cv2.line(frame, (left_part - 340, horizontal_line), (right_part + 320, horizontal_line), (0, 255, 255), 8)
        frame = cv2.line(frame, (left_part, horizontal_line - 175), (left_part, horizontal_line + 165), (0, 255, 255), 8)
        frame = cv2.line(frame, (right_part, horizontal_line - 175), (right_part, horizontal_line + 165), (0, 255, 255), 8)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("penalty_name", type=str)
    args = parser.parse_args()

    penalty_folder = os.path.join("dataset_fifa", args.penalty_name)
    combined_results_csv = os.path.join("dataset_fifa", "combined_results.csv")
    video_path = os.path.join("Penalty_videos", f"{args.penalty_name}.mp4")

    model_path = "./best_model.pth"
    model = load_model(model_path)
    input_data = preprocess_input(penalty_folder)

    with torch.no_grad():
        output = model(input_data)
        _, predicted_label_idx = torch.max(output, 1)

    label_mapping = {0: 'TR', 1: 'TL', 2: 'BR', 3: 'BL'}
    predicted_result = label_mapping[predicted_label_idx.item()]
    print(f"Predicted Result: {predicted_result}")

    true_label = get_true_label(penalty_folder, combined_results_csv)
    print(f"True Label: {true_label}")

    output_video_path = f"prediction_{args.penalty_name}.mp4"
    draw_prediction(video_path, predicted_result, output_video_path)

if __name__ == "__main__":
    main()