**I) Dataset collection**

  For the initial dataset, it is crucial to ensure that we have videos (ideally 30fps) that last exactly 1 second (to have exactly 30 frames per video) with the camera angle behind the shooter. However, if the videos do not have 30 frames (e.g., 25 frames), it is not an issue as long as they represent the 1 second before the shot because later, we can extend them to 30 frames. The goal is to collect high-quality videos from FIFA or real-life scenarios. It is essential that the video stops as late as possible before the kick to ensure we only process the run-up before the shot.

**A) Videos from real-life scenarios**

**B) Videos from FIFA **

First of all, we need to download the videos from YouTube. And here are two channels having a lot of penalty videos :
- https://www.youtube.com/@fifagrg
- https://www.youtube.com/@ALLGOAL

Now that the videos are downloaded from YouTube under high quality, we need to detect each penalty and create two videos out of each penalty. One video representing the kicker right before the contact of the ball. This video is 1s long and constituted of 30 frames. As well as one video of 0.7s that will be representing the result of the penalty where we will apply some pose estimation on the ball to obtain the true penalty result.

  **Step 1: Detect Frames**

First, run the detect.py script to identify the frames where the kicker touches the ball. This script requires the weights for the detection model and the video file to be processed. Here, for the example, we will process the first video that is called shoot-1, and we will use the heaviest model found on YOLOv9 :
   ```
   python detect.py --weights yolov9-e.pt --conf 0.2 --source shoot-1.mp4
```
  

This will output a CSV file named detected_frames_1.csv, which contains the frame numbers where the kicker is in contact with the ball for the video 1.
  
**  Step 2: Cut Videos**

Next, run the video_cut.py script to generate short video clips around the moment of ball contact. This script takes the detected_frames_1.csv file as input and produces video clips for each penalty. More specifically, it will output the folder Videos_CUT_1, where all the videos before ball contact will be stored, and a folder called Videos_RESULT_1, where all the clips containing the result of the penalty will be stored :
  python videocut.py --source shoot-1.mp4

The videos found in the Videos_CUT_1 folder will be directly used for the body joints coordinate extraction, while the ones found in the Videos_RESULT_1 folder needs to be processed a bit further.

**  Step 3: Run Pose Estimation on the ball**

Use the run_detection.py script to execute the detection.py and to apply pose estimation on the ball in order to determine the outcome of each penalty. This script requires the confidence threshold for detection and the folder containing the video clips :
  python run_detection.py --conf 0.6 --source-folder Videos_RESULT_1

This will output the folder Detection_RESULT_1, where inside, for each video a ball_tracking.csv file is containing the estimated position of the ball for each frame, as well as a video clip with the visualization of the estimated ball position thanks to a Kalman Filter. 

**  Step 4: Obtain all the results from one video**

Finally, run the results.py file script to combine all the results of the penalties from one video. And this will output the results_1.csv file where all the penalties results will be stored :
  python results.py --source-folder Detection_RESULT_1

**  Step 5: Concatenate Results**

The final step is to concatenate all the results from all the different videos and this is done by running the script combine.py that will concatenate all the penalties in the right order following the number of each folder :
  python combine.py

**II) Body joint coordinates Extraction**
  
  Once we have all our videos in a folder, we need to apply "Pose_estimation.ipynb" by modifying the code to point to the folder containing the videos and an output folder. The code will process all videos in the folder one by one and create, in the output folder, a new folder for each video containing CSV files for each frame. Each CSV file will describe the positions of the keypoints [left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle] in this order, along with their certainty scores.
  
  For example, for an input folder containing 10 videos of 30 frames each, we will have an output folder with 10 folders, each containing 30 CSV files with 12 rows and 3 columns ('x': x position, 'y': y position, 'z': certainty score).
  
  Once the body joints are extracted, we need to use "CSV_zeros.ipynb" to handle zeros. Occasionally, the model used for pose estimation does not detect certain body parts that may be obscured, setting 'x' and 'y' to 0.0 but not the certainty score (which is just very low). The "CSV_zeros.ipynb" script addresses this issue by setting the certainty scores of keypoints positioned at [0.0, 0.0] to zero.
  
  Finally, if the original videos were not at 30fps (e.g., 25fps), we need to use the "25_to_30_fps.ipynb" script on the output folder. This algorithm uses interpolation techniques to extend a folder of 25 CSV files to 30 CSV files.
  
  After completing all these steps, the final dataset will be ready for model processing. Additionally, you need to provide a CSV file with a table containing 2 columns and m+1 rows (where m represents the number of videos). The first column should contain the names of the results, and the second column should contain the labels (TR: Top Right, TC: Top Center, TL: Top Left, BR: Bottom Right, BC: Bottom Center, BL: Bottom Left).



For helping, here is an initial dataset containing 123 real-life videos of 25 frames each : https://drive.google.com/drive/folders/1RcVAFkH2hxDxp-5_n5XqajuaeXmDl0Yp?usp=sharing
The file "penalty_16.avi" is a visualization of the Body joint extraction of the video "penalty_16" of this dataset.
And "Real_Dataset.zip" is the final version dataset that the model takes as input.



**II) Model **
