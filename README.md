# Overview

This project aims to predict the direction of a penalty kick based on the player's run-up. We use a combination of video data from FIFA video games and real-life footage to develop and train a pose estimation model that can accurately analyze the player's movements and predict the kick's direction.



## I) Dataset collection

  For the initial dataset, it is crucial to ensure that we have videos (ideally 30fps) that last exactly 1 second (to have exactly 30 frames per video) with the camera angle behind the shooter. We chose videos at 30fps to facilitate the search for real-life videos on the internet, as this is a common frame rate for high-quality recordings. However, if the videos do not have 30 frames (e.g., 25 frames), it is not an issue as long as they represent the 1 second before the shot because later, we can extend them to 30 frames.

The goal is to collect high-quality videos from FIFA or real-life scenarios. We selected 1 second before the shot because we estimated that this duration is sufficient to capture the run-up without being excessively long, which could be inconsistent. Additionally, we chose the camera angle from behind the shooter as it is the most representative of the run-up and would facilitate the collection of FIFA videos. It is essential that the video stops as late as possible before the kick to ensure we only process the run-up before the shot.

### A) Videos from real-life scenarios

For the videos of real life scenarios, we manually selected one by one the penalties by downloading them from YouTube and then cutting them at the right length. This was done manually for all the videos. This dataset is composed of 123 penalties but after vertical flippling and x-translation, and both at the same time, the dataset is quadrupled : **492 penalties**.

### B) Videos from FIFA

First, we need to download the videos from YouTube. Here are two channels with many penalty videos:

- https://www.youtube.com/@fifagrg
- https://www.youtube.com/@ALLGOAL

After downloading the videos in high quality, we detect each penalty and create two videos from each penalty: one representing the kicker right before the contact with the ball (1s long, 30 frames) and another representing the result of the penalty (0.7s long).

All the files are contained in the dataset FIFA folder and it goes without mention that the paths in each file needs to be adjusted to the specific path of the video files. This dataset is composed of 1'385 penalties (25 youtube videos processed from "FIFA GRG"'s channel) but after vertical flippling and x-translation, and both at the same time, the dataset is quadrupledd : **5'540 penalties**.


## II) Body joint coordinates Extraction
  
  Once we have all our videos in a folder, we need to apply "Pose_estimation.ipynb" by modifying the code to point to the folder containing the videos and an output folder. The code will process all videos in the folder one by one and create, in the output folder, a new folder for each video containing CSV files for each frame. Each CSV file will describe the positions of the keypoints [left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle] in this order, along with their certainty scores.
  
  For example, for an input folder containing 10 videos of 30 frames each, we will have an output folder with 10 folders, each containing 30 CSV files with 12 rows and 3 columns ('x': x position, 'y': y position, 'z': certainty score).
  
  Once the body joints are extracted, we need to use "CSV_zeros.ipynb" to handle zeros. Occasionally, the model used for pose estimation does not detect certain body parts that may be obscured, setting 'x' and 'y' to 0.0 but not the certainty score (which is just very low). The "CSV_zeros.ipynb" script addresses this issue by setting the certainty scores of keypoints positioned at [0.0, 0.0] to zero.
  
  Finally, if the original videos were not at 30fps (e.g., 25fps), we need to use the "25_to_30_fps.ipynb" script on the output folder. This algorithm uses interpolation techniques to extend a folder of 25 CSV files to 30 CSV files.
  
  After completing all these steps, the final dataset will be ready for model processing. Additionally, you need to provide a CSV file with a table containing 2 columns and m+1 rows (where m represents the number of videos). The first column should contain the names of the results, and the second column should contain the labels (TR: Top Right, TC: Top Center, TL: Top Left, BR: Bottom Right, BC: Bottom Center, BL: Bottom Left).

For helping, here is an initial dataset containing 123 real-life videos of 25 frames each : - https://drive.google.com/drive/folders/1RcVAFkH2hxDxp-5_n5XqajuaeXmDl0Yp?usp=sharing

The file "penalty_16.avi" is a visualization of the Body joint extraction of the video "penalty_16" of this dataset. And the final dataset is in "datasets".

All the files are contained in the directory Body_Joint_Extraction.

# Usage

### Step 1: Detect Frames

Run the detect.py script to identify the frames where the kicker touches the ball. This script requires the weights for the detection model and the video file to be processed.
   ```
   python detect.py --weights yolov9-e.pt --conf 0.2 --source shoot-1.mp4
```
  
This will output a CSV file named detected_frames_1.csv, containing the frame numbers where the kicker contacts the ball for video 1.

### Step 2: Cut Videos

Run the video_cut.py script to generate short video clips around the moment of ball contact.

```
  python videocut.py --source shoot-1.mp4
```
This will create two folders: Videos_CUT_1 (videos before ball contact) and Videos_RESULT_1 (videos with the penalty result).


### Step 3: Run Pose Estimation on the ball

Use the run_detection.py script to apply pose estimation on the ball and determine the outcome of each penalty.

```
  python run_detection.py --conf 0.6 --source-folder Videos_RESULT_1
```
This will output a folder Detection_RESULT_1, containing ball_tracking.csv files with the estimated ball positions and video clips with visualizations of the estimated positions.

### Step 4: Obtain all the results from one video

Run the results.py script to combine all the results of the penalties from one video.

```
  python results.py --source-folder Detection_RESULT_1
```
This will produce a results_1.csv file with all the penalty results.

### Step 5: Concatenate Results

Run the combine.py script to concatenate all the results from the different videos.

```
  python combine.py
```
Ensure all paths in each file are adjusted to the specific path of the video files. The FIFA dataset is composed of 1,385 penalties (from 25 YouTube videos), and after augmentation, it is quadrupled to 5,540 penalties.

### Body Joint Coordinates Extraction
1. Apply "Pose_estimation.ipynb" to the videos folder, specifying input and output folders.
2. Use "CSV_zeros.ipynb" to handle zeros in the data.
3. If necessary, run "25_to_30_fps.ipynb" to convert 25fps videos to 30fps.

### Model


## III) Model

We will now look at the model and how to use it. Regarding the datasets that we are using as inputs, we have two datasets : dataset_fifa and dataset_real. Both can be found in the Datasets folder.

### Step 1: Pre-process Data

To prepare the data for model training, run the preprocess.py script. The inputs include folders containing penalty data, where each penalty folder has 30 csv files (for the 30 frames) with the 12 body joints coordinates and their certainty score. As well as the combined_results.csv file containing the label for each penalty. This will output the .pt files for training, validation, and testing : train_data, val_data and test_data. Here the data has been quadrupled by first doing a vertical flipping, inversing the labels, and both at the same time, on the normalized coordinates :
```
python preprocess.py
```

### Step 2: Train the Model

Run the train.py script to train the model. This script will execute the sttrans.py script, displaying validation and training loss in the terminal and as a graph. It also compares true results with predictions for the testing data and prints the model accuracy. The trained model is saved as best_model.pth.
```
python training.py
```


