You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_Apr_2022_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the April edition of the 2022 Tabular Playground Series! This month's challenge is a time series classification problem.

You've been provided with thousands of sixty-second sequences of biological sensor data recorded from several hundred participants who could have been in either of two possible activity states. Can you determine what state a participant was in from the sensor data?

About the Tabular Playground Series

Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly.
The goal of these competitions is to provide a fun and approachable-for-anyone tabular dataset to model. These competitions are a great choice for people looking for something in between the Titanic Getting Started competition and the Featured competitions. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you; thus, we encourage you to avoid saturating the leaderboard.
For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals.

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

Submission File

For each sequence in the test set, you must predict a probability for the state variable. The file should contain a header and have the following format:

    sequence,state
    25968,0
    25969,0
    25970,0
    ...


##  Dataset Description:
In this competition, you'll classify 60-second sequences of sensor data, indicating whether a subject was in either of two activity states for the duration of the sequence.
Files and Field Descriptions

train.csv - the training set, comprising ~26,000 60-second recordings of thirteen biological sensors for almost one thousand experimental participants

    sequence - a unique id for each sequence
    subject - a unique id for the subject in the experiment
    step - time step of the recording, in one second intervals
    sensor_00 - sensor_12 - the value for each of the thirteen sensors at that time step

train_labels.csv - the class label for each sequence. 

    sequence - the unique id for each sequence.
    state - the state associated to each sequence. This is the target which you are trying to predict.

test.csv - the test set. For each of the ~12,000 sequences, you should predict a value for that sequence's state.

sample_submission.csv - a sample submission file in the correct format.

train.csv - column name: sequence, subject, step, sensor_00, sensor_01, sensor_02, sensor_03, sensor_04, sensor_05, sensor_06, sensor_07, sensor_08, sensor_09, sensor_10, sensor_11, sensor_12
test.csv - column name: sequence, subject, step, sensor_00, sensor_01, sensor_02, sensor_03, sensor_04, sensor_05, sensor_06, sensor_07, sensor_08, sensor_09, sensor_10, sensor_11, sensor_12
train_labels.csv - column name: sequence, state


## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-apr-2022. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv, train_labels.csv

## Solution Description:
My solution was similar to most of the ones I have seen - an ensemble of RNN and gradient boost solutions.

The main difference between mine was the architecture of the RNN model. I first projected each sequence into an additional 16 dimensions with a linear dense layer - so my data was (n, 60, 13, 16) dimensions. Then I applied a separate GRU network with 4 GRU layers to each of the 13 sequences separately. This stopped the GRU model from overfitting to noisy covariates between each sequence, and allowed the network to converge better. This model performed 0.9839 on the private LB and 0.985 on the public LB.

Ensembling this model with XGBoost, 1D-convolutional, and public LSTM models achieved my final private LB score of 0.98797.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: