You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Classification_with_an_Academic_Success_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
OverviewWelcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: The goal of this competition is to predict academic risk of students in higher education.


##  Evaluation Metric:
Submissions are evaluated using the accuracy score.

Submission File
For each id row in the test set, you must predict the class value of the Target, which is a categorical academic risk assessment. The file should contain a header and have the following format:

    id,Target
    76518,Graduate
    76519,Graduate
    76520,Graduate
    etc.


##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Predict Students' Dropout and Academic Success dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance. Please refer to the original dataset for feature feature explanations.

Files

    train.csv - the training dataset; Target is the categorical target
    test.csv - the test dataset; your objective is to predict the class of Target for each row
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, Marital status, Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Nacionality, Mother's qualification, Father's qualification, Mother's occupation, Father's occupation, Admission grade, Displaced, Educational special needs, Debtor, Tuition fees up to date, Gender, Scholarship holder, Age at enrollment, International, Curricular units 1st sem (credited), Curricular units 1st sem (enrolled), Curricular units 1st sem (evaluations), Curricular units 1st sem (approved), Curricular units 1st sem (grade), Curricular units 1st sem (without evaluations), Curricular units 2nd sem (credited), Curricular units 2nd sem (enrolled), Curricular units 2nd sem (evaluations), Curricular units 2nd sem (approved), Curricular units 2nd sem (grade), Curricular units 2nd sem (without evaluations), Unemployment rate, Inflation rate, GDP, Target
test.csv - column name: id, Marital status, Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Nacionality, Mother's qualification, Father's qualification, Mother's occupation, Father's occupation, Admission grade, Displaced, Educational special needs, Debtor, Tuition fees up to date, Gender, Scholarship holder, Age at enrollment, International, Curricular units 1st sem (credited), Curricular units 1st sem (enrolled), Curricular units 1st sem (evaluations), Curricular units 1st sem (approved), Curricular units 1st sem (grade), Curricular units 1st sem (without evaluations), Curricular units 2nd sem (credited), Curricular units 2nd sem (enrolled), Curricular units 2nd sem (evaluations), Curricular units 2nd sem (approved), Curricular units 2nd sem (grade), Curricular units 2nd sem (without evaluations), Unemployment rate, Inflation rate, GDP


## Dataset folder Location: 
../../kaggle-data/playground-series-s4e6. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Hi everyone,

My solution is simple:

    5-fold StratifiedKFold for validation.
    Included the original dataset.
    Ensemble of XGB and LGBM.
    Optuna hyperparamter tuning.

 I didn't dedicate as much time to this competition as I usually would because I noticed a significant amount of randomness in the outcomes. My assessment appears to be validated due to a new Kaggler @furgalhachaimajhi achieving 2nd place (congratulations btw!) with only a single submission.

The most interesting aspect of this competition for me was that single models or very small ensembles seemed to perform exceptionally well. It reminds of a Playground competition over a year ago where I achieved 54th place with a single CatBoost model. Single Model Catboost PS S3 E4

My solution was heavily inspired by @rzatemizel's notebook LGBM + CATB+ XGB + NN: Voting or Stacking? (make sure to upvote) which employs recursive feature elimination with cross-validation to determine the models to include in the ensemble..


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: