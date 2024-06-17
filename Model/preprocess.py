import torch
import os
import numpy as np
import pandas as pd

class PenaltyDataset(torch.utils.data.Dataset):
    def __init__(self, penalty_folder, combined_results_csv, label_mapping):
        self.penalty_folder = penalty_folder
        self.combined_results_csv = combined_results_csv
        self.frames = []
        self.labels = []
        self.folders = []  
        self.label_mapping = label_mapping
        self.load_data()

    def load_data(self):
        combined_results = pd.read_csv(self.combined_results_csv)
        combined_results['Folder'] = combined_results['Video Name'].apply(lambda x: x.replace('.mp4', ''))
        folder_to_label = dict(zip(combined_results['Folder'], combined_results['Zone']))

        expected_length = 36
        penalty_dirs = sorted(os.listdir(self.penalty_folder))
        
        for penalty_dir in penalty_dirs:
            penalty_path = os.path.join(self.penalty_folder, penalty_dir)
            if os.path.isdir(penalty_path):
                frames = []
                for I in range(30):
                    frame_file = os.path.join(penalty_path, f'keypoints_frame_{I}.csv')
                    if os.path.exists(frame_file):
                        frame_data = np.genfromtxt(frame_file, delimiter=',', skip_header=1)
                        if frame_data.ndim == 1:
                            frame_data = np.expand_dims(frame_data, axis=0)
                        if frame_data.size == 0 or frame_data.shape[1] != 4:
                            break
                        frame_data = frame_data[:, 1:]  
                        frame_data = frame_data.reshape(-1)
                        if frame_data.shape[0] < expected_length:
                            frame_data = np.pad(frame_data, (0, expected_length - frame_data.shape[0]), 'constant')

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
                
                if len(frames) == 30:
                    frames = np.array(frames)
                    folder_name = penalty_dir
                    if folder_name in folder_to_label:
                        label_name = folder_to_label[folder_name]
                        if label_name in self.label_mapping:
                            if label_name not in ['TC', 'BC']:  
                                self.frames.append(frames)
                                self.labels.append(self.label_mapping[label_name])
                                self.folders.append(folder_name) 
                    else:
                        pass

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.frames[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.folders[idx]
        )

def flip_coordinates_and_labels(data, labels):
    flipped_data = data.clone()
    flipped_data[:, :, 0::3] = 1 - flipped_data[:, :, 0::3]  
    flipped_data[:, :, 1::3] = 1 - flipped_data[:, :, 1::3]
    
    label_mapping = {0: 1, 1: 0, 2: 3, 3: 2}
    flipped_labels = torch.tensor([label_mapping[label.item()] for label in labels])
    
    return flipped_data, flipped_labels

def shift_coordinates(data, shift_value=0.05):
    shifted_data = data.clone()
    shifted_data[:, :, 0::3] = torch.clamp(shifted_data[:, :, 0::3] + shift_value, 0, 1)  
    return shifted_data

def augment_data(data, labels, folders):
    flipped_data, flipped_labels = flip_coordinates_and_labels(data, labels)
    shifted_data = shift_coordinates(data)
    shifted_flipped_data = shift_coordinates(flipped_data)

    augmented_data = torch.cat((data, flipped_data, shifted_data, shifted_flipped_data), dim=0)
    augmented_labels = torch.cat((labels, flipped_labels, labels, flipped_labels), dim=0)
    augmented_folders = folders + [folder + '_flipped' for folder in folders] + \
                        [folder + '_shifted' for folder in folders] + \
                        [folder + '_flipped_shifted' for folder in folders]

    return augmented_data, augmented_labels, augmented_folders

def main():
    penalty_folder = 'dataset_fifa'
    combined_results_csv = 'dataset_fifa/combined_results.csv'
    label_mapping = {'TR': 0, 'TL': 1, 'BR': 2, 'BL': 3} 
    
    dataset = PenaltyDataset(penalty_folder, combined_results_csv, label_mapping)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    all_data = []
    all_labels = []
    all_folders = []
    
    for data, labels, folders in dataloader:
        all_data.append(data)
        all_labels.append(labels)
        all_folders.extend(folders)
    
    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)
    
    all_data, all_labels, all_folders = augment_data(all_data, all_labels, all_folders)
    
    total_data_points = all_data.shape[0]
    
    train_size = int(0.8 * total_data_points)
    val_size = int(0.1 * total_data_points)
    test_size = total_data_points - train_size - val_size
    
    indices = list(range(total_data_points))
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]
    
    print(f"Total penalties: {total_data_points}")
    print(f"Train penalties: {train_size}, Validation penalties: {val_size}, Test penalties: {test_size}")
    
    train_data = all_data[train_indices], all_labels[train_indices]
    val_data = all_data[val_indices], all_labels[val_indices]
    test_data = all_data[test_indices], all_labels[test_indices]
    
    train_folders = [all_folders[I] for I in train_indices]
    val_folders = [all_folders[I] for I in val_indices]
    test_folders = [all_folders[I] for I in test_indices]
    
    torch.save((train_data, train_folders), 'train_data.pt')
    torch.save((val_data, val_folders), 'val_data.pt')
    torch.save((test_data, test_folders), 'test_data.pt')
    
if __name__ == "__main__":
    main()
