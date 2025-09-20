You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named CareerCon_2019_-_Help_Navigate_Robots_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
CareerCon 2019 is upon us!
CareerCon is a digital event all about landing your first data science job — and registration is now open! Ahead of the event, we have a fun competition to get you started. See below for a unique challenge and opportunity to share your resume with select CareerCon sponsors.

The Competition

Robots are smart… by design. To fully understand and properly navigate a task, however, they need input about their environment.

In this competition, you’ll help robots recognize the floor surface they’re standing on using data collected from Inertial Measurement Units (IMU sensors).

We’ve collected IMU sensor data while driving a small mobile robot over different floor surfaces on the university premises. The task is to predict which one of the nine floor types (carpet, tiles, concrete) the robot is on using sensor data such as acceleration and velocity.
Succeed and you'll help improve the navigation of robots without assistance across many different surfaces, so they won’t fall down on the job.


##  Evaluation Metric:
Submissions are evaluated on Multiclass Accuracy, which is simply the average number of observations with the correct label.

Submission File

For each series_id in the test set, you must predict a value for the surface variable. The file should have the following format:

    series_id,surface
    0,fine_concrete
    1,concrete
    2,concrete
    etc.


##  Dataset Description:
X_[train/test].csv - the input data, covering 10 sensor channels and 128 measurements per time series plus three ID columns:

    -row_id: The ID for this row.
    -series_id: ID number for the measurement series. Foreign key to y_train/sample_submission.
    -measurement_number: Measurement number within the series.

The orientation channels encode the current angles how the robot is oriented as a quaternion (see Wikipedia). Angular velocity describes the angle and speed of motion, and linear acceleration components describe how the speed is changing at different times. The 10 sensor channels are:

    orientation_X
    orientation_Y
    orientation_Z
    orientation_W
    angular_velocity_X
    angular_velocity_Y
    angular_velocity_Z
    linear_acceleration_X
    linear_acceleration_Y
    linear_acceleration_Z


y_train.csv - the surfaces for training set.

    -series_id: ID number for the measurement series.
    -group_id: ID number for all of the measurements taken in a recording session. Provided for the training set only, to enable more cross validation strategies.
    -surface: the target for this competition.

sample_submission.csv - a sample submission file in the correct format.

X_test.csv - column name: row_id, series_id, measurement_number, orientation_X, orientation_Y, orientation_Z, orientation_W, angular_velocity_X, angular_velocity_Y, angular_velocity_Z, linear_acceleration_X, linear_acceleration_Y, linear_acceleration_Z
X_train.csv - column name: row_id, series_id, measurement_number, orientation_X, orientation_Y, orientation_Z, orientation_W, angular_velocity_X, angular_velocity_Y, angular_velocity_Z, linear_acceleration_X, linear_acceleration_Y, linear_acceleration_Z
y_train.csv - column name: series_id, group_id, surface


## Dataset folder Location: 
../../kaggle-data/career-con-2019. In this folder, there are the following files you can use: X_test.csv, X_train.csv, y_train.csv, sample_submission.csv

## Solution Description:
My method was pretty straight forward and based on 1-D Convolutional NNs.

Features:
I used the Accel, Gyro features as it is. Using raw orientation features did not make sense to me, so I used the first order difference of the orientation features (after converting them to Euler angles), so that any vibrations caused by various surfaces are captured.
Also used FFT of Accel/Gyro measurements as features.

CV:

As discussed by multiple people on the forums, GroupKFold makes sense here. k=3 gave me ~0.65-0.7 CV scores which translated to ~0.8 on the Public leaderboard.

Model:

1D Convolution based NN. I saw slightly better results with the SeparableConv1D implementation in Keras, but I'm sure the simpler Conv1D layer if tuned properly should perform equally well.

the architecture of the model is as follows:


The model takes two types of inputs:
	1.	Time-domain features (inputs_t): Shape is (128, len(feat_cols)), where 128 could be the number of time steps, and len(feat_cols) is the number of features at each time step.
	2.	Frequency-domain features (inputs_f): Shape is (feat_fft_array.shape[1], feat_fft_array.shape[2]), corresponding to FFT magnitude features.

The model processes both inputs separately with convolutional layers and then combines the learned features before producing the final predictions.

Detailed Architecture

Time-domain Branch (inputs_t):
	1.	SeparableConv1D Layer: The first layer applies a depthwise separable 1D convolution with 32 filters, a kernel size of 8, and a stride of 2. It uses ReLU activation and L2 regularization (if kr is set).
	2.	Dropout Layer: Dropout is applied after each convolutional layer to prevent overfitting.
	3.	SeparableConv1D Layer: The second convolution layer has 64 filters, kernel size of 8, and stride of 4, followed by dropout.
	4.	SeparableConv1D Layer: The third convolution layer increases the number of filters to 128, with the same kernel size and stride.
	5.	SeparableConv1D Layer: The fourth convolutional layer further increases the filters to 256.
	6.	Reshape Layer: The output of the last convolution is reshaped to a flat vector of size 256.
	7.	Dense Layer: A fully connected layer with 64 units and ReLU activation follows.
	8.	Dense Layer: Another fully connected layer with 64 units and ReLU activation, producing the final feature representation for the time-domain input.

Frequency-domain Branch (inputs_f):
	1.	The frequency-domain input undergoes similar processing:
	•	SeparableConv1D Layers: Four separable convolution layers with different filter sizes (32, 64, 128, 256) and kernel sizes of 8, all followed by dropout.
	•	Reshape Layer: After the final convolution, the output is reshaped to a flat vector of size 256.
	•	Dense Layers: The frequency-domain features are then processed by two fully connected layers with 64 units and ReLU activation.

Combining Time and Frequency Features:
	•	Concatenate Layer: The outputs of the time-domain and frequency-domain branches are concatenated along the feature axis, creating a combined feature vector.
	•	Dense Layer: A fully connected layer with 64 units is applied to the concatenated feature vector.
	•	Prediction Layer: The final layer is a softmax output layer with num_surfaces units, where num_surfaces is the number of unique surface classes in the target data.

Model Compilation
	•	Optimizer: Adam optimizer is used for training.
	•	Loss Function: Sparse categorical crossentropy is chosen as the loss function, which is appropriate for multi-class classification tasks where labels are provided as integers.
	•	Metrics: Accuracy is tracked during training.
    
I have shared the kernel with the Starter code for my solution. This kernel does not do any chaining of test samples and by itself should be able to score ~0.87 on the Private Leaderboard. Its not well commented at this time, but I will add comments to it.

On top of it, I used a simple chaining mechanism to stretch the leaderboard score to 0.99 (In my opinion, this does not add much value in practice, and I only did it to stay relevant on the leaderboard. No matter how good your chaining algorithm is, all you have to do in the real world is take some additional steps)

Hope the models resulting from this contest can be put to practical use by the parties who shared the data.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: