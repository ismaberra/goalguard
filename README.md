# Overview

This project aims to predict the direction of a penalty kick based on the player's run-up. We use a combination of video data from FIFA video games and real-life footage to develop and train a pose estimation model that can accurately analyze the player's movements and predict the kick's direction.


It is an "extension" of this project : [Penalty_Kick_Analysis.pdf](https://github.com/user-attachments/files/15859660/Penalty_Kick_Analysis.pdf)

# I) Dataset collection

  For the initial dataset, it is crucial to ensure that we have videos (ideally 30fps) that last exactly 1 second (to have exactly 30 frames per video) with the camera angle behind the shooter. We choose videos at 30fps to facilitate the search for real-life videos on the internet, as this is a common frame rate for high-quality recordings. However, if the videos do not have 30 frames (e.g., 25 frames), it is not an issue as long as they represent the 1 second before the shot because later, we can extend them to 30 frames.

The goal is to collect high-quality videos from FIFA or real-life scenarios. We selected 1 second before the shot because we estimated that this duration is sufficient to capture the run-up without being excessively long, which could be inconsistent. Additionally, we chose the camera angle from behind the shooter as it is the most representative of the run-up and would facilitate the collection of FIFA videos. It is essential that the video stops as late as possible before the kick to ensure we only process the run-up before the shot.

### A) Videos from real-life scenarios

For the videos of real-life scenarios, we manually selected each penalty by downloading them from YouTube and then cutting them to the right length. This was done manually for all the videos. This dataset is composed of 123 penalties. Collecting these videos was very challenging because football organizations such as UEFA, FIFA, etc., are very strict about access to match footage on the internet. To access these matches, one would typically need to pay for subscriptions to get special access to match videos. Additionally, the camera angle from behind the shooter has only been available for a few years, thanks to technologies like drone usage for filming sequences.


The penalties collected are from the few replays available on the official YouTube channels of these organizations. We downloaded the videos in 25fps since these channels only broadcasted them at that frame rate. Then, we manually cut the videos using iMovie because the video quality was not good enough to ensure that an automated model could perform this task accurately. 

Here is the initial dataset containing 123 real-life videos of 25 frames each :
- https://drive.google.com/drive/folders/1RcVAFkH2hxDxp-5_n5XqajuaeXmDl0Yp?usp=sharing

### B) Videos from FIFA


First of all, we need to download the videos from YouTube. And here are two channels having a lot of penalty videos :

1. https://www.youtube.com/@fifagrg
2. https://www.youtube.com/@ALLGOAL

Now that the videos are downloaded from YouTube under high quality, we need to detect each penalty and create two videos out of each penalty. One video representing the kicker right before the contact of the ball. This video is 1s long and constituted of 30 frames. As well as one video of 0.7s that will be representing the result of the penalty where we will apply some pose estimation on the ball to obtain the true penalty result.

#### Step 1: Detect Frames

First, run the detect.py script to identify the frames where the kicker touches the ball. This script requires the weights for the detection model and the video file to be processed. Here, for the example, we will process the first video that is called shoot-1, and we will use the heaviest model found on YOLOv9 :

   ```
   python detect.py --weights yolov9-e.pt --conf 0.2 --source shoot-1.mp4
```

This will output a CSV file named detected_frames_1.csv, which contains the frame numbers where the kicker is in contact with the ball for the video 1.

#### Step 2: Cut Videos

Next, run the video_cut.py script to generate short video clips around the moment of ball contact. This script takes the detected_frames_1.csv file as input and produces video clips for each penalty. More specifically, it will output the folder Videos_CUT_1, where all the videos before ball contact will be stored, and a folder called Videos_RESULT_1, where all the clips containing the result of the penalty will be stored :

```
  python videocut.py --source shoot-1.mp4
```
The videos found in the Videos_CUT_1 folder will be directly used for the body joints coordinate extraction, while the ones found in the Videos_RESULT_1 folder needs to be processed a bit further.

#### Step 3: Run Pose Estimation on the ball

Use the run_detection.py script to execute the detection.py and to apply pose estimation on the ball in order to determine the outcome of each penalty. This script requires the confidence threshold for detection and the folder containing the video clips :

```
  python run_detection.py --conf 0.6 --source-folder Videos_RESULT_1
```
This will output the folder Detection_RESULT_1, where inside, for each video a ball_tracking.csv file is containing the estimated position of the ball for each frame, as well as a video clip with the visualization of the estimated ball position thanks to a Kalman Filter.

#### Step 4: Obtain all the results from one video

Finally, run the results.py file script to combine all the results of the penalties from one video. And this will output the results_1.csv file where all the penalties results will be stored :


```
  python results.py --source-folder Detection_RESULT_1
```
#### Step 5: Concatenate Results

The final step is to concatenate all the results from all the different videos and this is done by running the script combine.py that will concatenate all the penalties in the right order following the number of each folder :

```
  python combine.py
```
All the files are contained in the dataset FIFA folder and it goes without mention that the paths in each file needs to be adjusted to the specific path of the video files. This dataset is composed of 1'385 penalties (25 youtube videos processed from "FIFA GRG"'s channel).


# II) Body joint coordinates Extraction
  
  Once we have all our videos in a folder, we need to apply "Pose_estimation.ipynb" by modifying the code to point to the folder containing the videos and an output folder. The code will process all videos in the folder one by one and create, in the output folder, a new folder for each video containing one CSV file per frame. Each CSV file will describe the positions of the keypoints [left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left ankle, right ankle] in this order, along with their certainty scores (we did not take the head into account because the model does not detect the back of the head (only the nose, eyes and ears) and in any case we thought that it was not necessarily relevant for the penalty kick direction estimation).

For example, for an input folder containing 10 videos of 30 frames each, we will have an output folder with 10 folders, each containing 30 CSV files with 12 rows and 3 columns ('x': x position, 'y': y position, 'z': certainty score).

We initially tested several pose estimation models, but they did not perform well at all. Just before the midterm, we opted to use YOLOv7 (https://github.com/WongKinYiu/yolov7) for pose estimation, but the results were not very convincing and were quite approximate. After the midterm, we switched to YOLOv8 from Ultralytics (https://github.com/ultralytics/ultralytics), which is the latest version of YOLO capable of pose estimation and offers several models :

<img width="700" alt="Capture d’écran 2024-06-16 à 20 41 04" src="https://github.com/vita-epfl/goalguard/assets/83677158/89ddd51c-b484-4c2c-b838-a47d9bb89591">

For the FIFA videos, we used the YOLOv8s-pose model because its performance was sufficient for these videos (which are of very high quality and very clear), and with more than 1300 videos to process, the processing time of this model was suitable (it takes more than 2.5 hours to process over 1300 videos with CPU).

For the real-life videos, we used the most complex model since the videos were of lower quality and there were only 123 of them, so we could afford to use a more complex model (which takes approximately 1.5 hours with CPU).

Here is an example of the pose-estimation on a real video :

https://github.com/vita-epfl/goalguard/assets/83677158/0fdaa3ba-9b09-4694-9bb6-016685af3419

Once the body joints are extracted, we need to use "CSV_zeros.ipynb" to handle zeros. Occasionally, the model used for pose estimation does not detect certain body parts that may be obscured, setting 'x' and 'y' to 0.0 but not the certainty score (which is just very low). The "CSV_zeros.ipynb" script addresses this issue by setting the certainty scores of keypoints positioned at [0.0, 0.0] to zero. We applied this algorithm to all the CSV files.

Here is an example of a zero, the row concerns the left hand that is hidden by the body of the player :

<img width="300" alt="Capture d’écran 2024-06-16 à 20 44 27" src="https://github.com/vita-epfl/goalguard/assets/83677158/34bfb08a-d693-4993-ba0b-d7a18311f8c1">
<img width="461" alt="Capture d’écran 2024-06-16 à 20 53 52" src="https://github.com/vita-epfl/goalguard/assets/83677158/10afdfc0-62e6-4c6f-9801-f38454b61c7e">

Finally, if the original videos were not at 30fps (e.g., 25fps), we need to use the "25_to_30_fps.ipynb" script on the output folder. This algorithm uses interpolation techniques to extend a folder of 25 CSV files to 30 CSV files. We applied this algorithm to the real-life videos.

After completing all these steps, the final dataset will be ready for model processing. Additionally, you need to provide a CSV file with a table containing 2 columns and m+1 rows (where m represents the number of videos). The first column should contain the names of the results, and the second column should contain the labels (TR: Top Right, TC: Top Center, TL: Top Left, BR: Bottom Right, BC: Bottom Center, BL: Bottom Left).

Here is a visualization of the player's race after using the interpolation algorithm:









https://github.com/vita-epfl/goalguard/assets/83677158/e19450b4-b5ca-4ec7-9876-432f6a994275


Our final datasets are in the folder Datasets, the real dataset is composed of 123 samples but after vertical flippling and x-translation, and both at the same time, the dataset is quadrupled : **492 samples**. And the FIFA game dataset is composed of 1'385 samples  but after vertical flippling and x-translation, and both at the same time, the dataset is quadrupled : **5'540 samples**.
So our final dataset is composed of **6032 samples**.

# III) Model

We will now look at the model and how to use it. Regarding the datasets that we are using as inputs, we have two datasets : dataset_fifa and dataset_real. Both can be found in the Datasets folder.

### Step 1: Pre-process Data

To prepare the data for model training, run the preprocess.py script. The inputs include folders containing penalty data, where each penalty folder has 30 csv files (for the 30 frames) with the 12 body joints coordinates and their certainty score. As well as the combined_results.csv file containing the label for each penalty. This will output the .pt files for training, validation, and testing: train_data, val_data and test_data. Here the data has been quadrupled by first doing a vertical flipping, inversing the labels, and both at the same time, on the normalized coordinates:
```
python preprocess.py
```

### Step 2: Train the Model

The model is based on a transformer architecture, which is known for its effectiveness in handling sequential data. Our model comprises the following components:
- **Transformer Encoder Layers**: The model consists of 4 transformer encoder layers. Each layer has 4 attention heads, which allow the model to focus on different parts of the input sequence simultaneously.
- **Residual Blocks**: Each encoder layer includes residual connections to help gradients flow through the network, which eases the training of deeper models.
- **Feedforward Network**: After the multi-head attention mechanism, each layer has a feedforward neural network with a dimensionality of 768.
- **Convolutional Layers**: The input sequences are initially processed with convolutional layers to extract higher-level features before feeding them into the transformer encoder.
- **Dropout**: A dropout rate of 0.38415372018572036 is used to prevent overfitting.

The goal of this architecture is to leverage the self-attention mechanism of transformers to capture the dependencies between different frames in the penalty kick sequence. 
Furthermore, to optimize the model, we tuned the hyperparamaters using Optuna over 50 trials. The best hyperparameters found were:

| Hyperparameter             | Value                  |
|----------------------------|------------------------|
| Batch size                 | 52                     |
| Dropout                    | 0.38415372018572036    |
| Learning rate              | 0.00018305748553194186 |
| Weight decay               | 0.005783491595599079   |
| Number of layers           | 4                      |
| Number of heads            | 4                      |
| Feedforward dimension      | 768                    |

Now, run the train.py script to train the model. This script will execute the sttrans.py script, displaying validation and training loss in the terminal and as a graph. It also compares true results with predictions for the testing data and prints the model accuracy. The trained model is saved as best_model.pth.
```
python training.py
```

After training we obtained this graph, illustrating both the training and validation loss over the epochs, as well as the model's accuracy on the unseen testing data.

![Training_Validation_Loss](https://github.com/vita-epfl/goalguard/assets/146441738/95d9f0cd-da85-4f97-b7ed-ed7fb57bad72)

The final accuracy achieved was slighty higher than 35%, indicating a 10% improvement over a random guess. 

### Step 3: Inference

To run inference using the trained model, use the inference.py script. This script loads the trained model and processes a single penalty to predict it's result. It prints the predicted and true labels, as well as a video of said penalty with a mark visualizing the predicted zone.
```
python inference.py penalty_X
```

### Example Predictions

#### Correct Prediction
- Penalty: 5

https://github.com/vita-epfl/goalguard/assets/146441738/de9d479f-7859-4b16-93af-c4331b20a170

#### Incorrect Prediction
- Penalty: 218

https://github.com/vita-epfl/goalguard/assets/146441738/2739f9f5-aaa4-440e-aba3-9e73e37abd10

### Data Augmentation

To handle the small dataset and avoid overfitting, we applied data augmentation by quadrupling the data. The augmentations included vertical flipping, horizontal translation, and both transformations combined. This resulted in significantly more data, improving the model's ability to generalize better on unseen data.

### Limitations and Potential Improvements

The model's performance is constrained by the limited size and imbalance of the dataset. Potential improvements could include:
- Data augmentation 
- 3D pose estimation instead of 2D pose
- More accurate pose estimation of both the kicker and the ball
- Integration of additional features like whether the player is right-footed or left-footed

By addressing these limitations, we can further enhance the model's accuracy.

