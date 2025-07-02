You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Regression_with_a_Crab_Age_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
OverviewWelcome to the 2023 Kaggle Playground Series! Thank you to everyone who participated in and contributed to Season 3 Playground Series so far! 
Your Goal: For this Episode of the Series, your task is to use regression to predict the age of crabs given physical attributes. Good luck!StartMay 30, 2023CloseJun 13, 2023

##  Evaluation Metric:
Evaluation
Submissions will be evaluated using Mean Absolute Error (MAE),

where each x_i represents the predicted target, y_i represents the ground truth, and n is the number of rows in the test set.

#### Submission File
For each id in the test set, you must predict the target Age. The file should contain a header and have the following format:

    id,yield
    74051,10.2
    74051,3.6
    74051,11.9
    etc.


##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Crab Age Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.
Note: You can use this notebook to generate additional synthetic data for this competition if you would like.

Files

    train.csv - the training dataset; Age is the target
    test.csv - the test dataset; your objective is to predict the probability of Age (the ground truth is int but you can predict int or float)
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, Sex, Length, Diameter, Height, Weight, Shucked Weight, Viscera Weight, Shell Weight, Age
test.csv - column name: id, Sex, Length, Diameter, Height, Weight, Shucked Weight, Viscera Weight, Shell Weight


## Dataset folder Location: 
../../kaggle-data/playground-series-s3e16. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:

Thanks to Kaggle for the challenge/ episode! It was a great experience to participate in a good regression challenge in this episode. Also, thanks to the fellow competitors for making this a memorable experience.
My approach is as below-

#### Data engineering

I considered the data generation notebook provided here as a base and generated a number of samples with varying parameters. I appended this additional synthetic data with the original data and used it collectively to train the model
I used the below additional features in the model otherwise. These features are mentioned in the below discussion post- https://www.kaggle.com/competitions/playground-series-s3e16/discussion/415721, many thanks to @pandeyg0811 for the efforts.

    a. Meat yield
    b. Surface Area
    c. Weight/ Shuck Weight
    d. Pseudo BMI
    e. Weight/ Length Squared
    f. Viscera Ratio

I also used log(1+x) transform on the weight column, but it did not help me greatly

#### Model strategy

1. I used a regression model approach rather than a classification approach using a stratified 5-fold CV. Validation was performed on the competition data only. CV correlated well with the public LB
2. I trained several candidate models with feature subsets, using the provided features and additional feature-subsets, from 6-10 features per model run. I trained several candidates using the same OOF structure, but different features per run.
3. I used tree models as alongside- XgBoost, LightGBM, Catboost, Gradient Boosting Machine, HistGradientBoostingRegressor as base learners
4. I used optuna and LAD regression to tune the predictions in a subsequent ensemble per run with(out) post-processing and discovered the favorable impact of post-processing (rounding to the nearest integer) on the CV score and LB score. Post-processing my predictions helped me improve my LB score considerably throughout the challenge
5. I did not post-process any predictions based on the training/ original data. Most of my models posited predictions in the range of 4-20 years only. I did not impute/ overlay any train/ original predictions on this data. Whenever I tried to do this, my score would decrease. Perhaps my method was plain and sub-optimal, while some creativity was needed herewith.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: