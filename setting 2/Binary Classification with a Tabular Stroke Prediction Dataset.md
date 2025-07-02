You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Binary_Classification_with_a_Tabular_Stroke_Prediction_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the 2023 edition of Kaggle's Playground Series!

Thank you to everyone who participated in and contributed to last year's Tabular Playground Series. And many thanks to all those who took the time to provide constructive feedback! We're thrilled that there continues to be interest in these types of challenges, and we're continuing the series this year but with a few changes.

First, the series is getting upgraded branding. We've dropped "Tabular" from the name because, while we anticipate this series will still have plenty of tabular competitions, we'll also be having some other formats as well. You'll also notice freshly-upgraded (better looking and more fun!) banner and thumbnail images. 

Second, rather than naming the challenges by month and year, we're moving to a Season-Edition format. This year is Season 3, and each challenge will be a new Edition. We're doing this to have more flexibility. Competitions going forward won't necessarily align with each month like they did in previous years (although some might!), we'll have competitions with different time durations, and we may have multiple competitions running at the same time on occasion.

Regardless of these changes, the goals of the Playground Series remain the same—to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. We hope we continue to meet this objective!

To start the year with some fun, January will be the month of Tabular Tuesday. We're launching four week-long tabular competitions, with each starting Tuesday 00:00 UTC. These will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

### Synthetically-Generated Datasets
Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

### Submission File
For each id in the test set, you must predict the probability for the target variable stroke. The file should contain a header and have the following format:

    id,stroke
    15304,0.23
    15305,0.55
    15306,0.98
    etc.


##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Stroke Prediction Dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

### Files

    train.csv - the training dataset; stroke is the binary target
    test.csv - the test dataset; your objective is to predict the probability of positive stroke
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke
test.csv - column name: id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status


## Dataset folder Location: 
../../kaggle-data/playground-series-s3e2. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Hi all,
It seems there was quite a good shake up given that dataset was highly imbalanced and AUC can vary a lot depending the number of samples. I realized there was a good difference between by OOF AUC and the leaderboard so I decided to trust only my CV (10 StratifiedKfold).

#### Tricks that worked

    Fill unknown category form smoking status as never smoked. The ituition was given on my EDA where you can see that unknown class has the lowest probability of stroke.
    Fill other class from gender as male. I spotted a boost on CV when filling that record in synthetic dataset. I didn't probe the leaderboard to validate this on test.
    Ensemble using gradient descent and ranking the predictions.
    Concat original stroke dataset and use StratifiedKfold where validation only has synthetic data.
    Feature selection using RecursiveFeatureElimanation. Additional features I tried:

        def generate_features(df):
            df['age/bmi'] = df.age / df.bmi
            df['age*bmi'] = df.age * df.bmi
            df['bmi/prime'] = df.bmi / 25
            df['obesity'] = df.avg_glucose_level * df.bmi / 1000
            df['blood_heart']= df.hypertension*df.heart_disease
            return df

#### Things that didn't work

Use forward selection taken from this notebook. This was my second submission and scored 0.89941 on private leaderboard. I think It didn't worked because the final ensemble was only composed of XGBoost models while my best submission has a wide variety of models.
MeanEncoder, WoEEncoder and CountFrequency encoder. Neither of those provided better solutions that OneHotEncoder.

#### Final Ensemble:
My final ensemble is composed of several models:

    LogisticRegression with RFE, l2, and liblinear solver.
    LogisticRegression with RFE, no regularization, lbfgs solver.
    LightGBM no RFE, no Feature Engineering.
    Another LightGBM with early stopping and monitoring logloss (yes, logloss no AUC).
    A Catboost model inspired in this notebook by @dmitryuarov. I made some modifications to make sure the OOF AUC was similar to the mean AUC by fold.
    A tuned XGBoost with feature engineering. (best single model) See the code and results replica Here

And that's all.
Many congratulations to the winners, looking forward to the next playground competitions..

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: