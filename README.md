I) Dataset and Body joint coordinates Extraction

  For the initial dataset, it is crucial to ensure that we have videos (ideally 30fps) that last exactly 1 second (to have exactly 30 frames per video). However, if the videos do not have 30 frames (e.g., 25 frames), it is not an issue as long as they represent the 1 second before the shot because later, we can extend them to 30 frames. The goal is to collect high-quality videos from FIFA or real-life scenarios. It is essential that the video stops as late as possible before the kick to ensure we only process the run-up before the shot.
  
  Once we have all our videos in a folder, we need to apply "Pose_estimation.ipynb" by modifying the code to point to the folder containing the videos and an output folder. The code will process all videos in the folder one by one and create, in the output folder, a new folder for each video containing CSV files for each frame. Each CSV file will describe the positions of the keypoints [left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle] in this order, along with their certainty scores.
  
  For example, for an input folder containing 10 videos of 30 frames each, we will have an output folder with 10 folders, each containing 30 CSV files with 12 rows and 3 columns ('x': x position, 'y': y position, 'z': certainty score).
  
  Once the body joints are extracted, we need to use "CSV_zeros.ipynb" to handle zeros. Occasionally, the model used for pose estimation does not detect certain body parts that may be obscured, setting 'x' and 'y' to 0.0 but not the certainty score (which is just very low). The "CSV_zeros.ipynb" script addresses this issue by setting the certainty scores of keypoints positioned at [0.0, 0.0] to zero.
  
  Finally, if the original videos were not at 30fps (e.g., 25fps), we need to use the "25_to_30_fps.ipynb" script on the output folder. This algorithm uses interpolation techniques to extend a folder of 25 CSV files to 30 CSV files.
  
  After completing all these steps, the final dataset will be ready for model processing. Additionally, you need to provide a CSV file with a table containing 2 columns and m+1 rows (where m represents the number of videos). The first column should contain the names of the results, and the second column should contain the labels (TR: Top Right, TC: Top Center, TL: Top Left, BR: Bottom Right, BC: Bottom Center, BL: Bottom Left).


II) Model 
