You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_Feb_2021_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly. 

In order to have a more consistent offering of these competitions for our community, we're trying a new experiment in 2021. We'll be launching month-long tabular Playground competitions on the 1st of every month and continue the experiment as long as there's sufficient interest and participation.

The goal of these competitions is to provide a fun, and approachable for anyone, tabular dataset. These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you. We encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals. 

The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting the amount of an insurance claim. Although the features are anonymized, they have properties relating to real-world features.


##  Evaluation Metric:
Submissions are scored on the root mean squared error. RMSE is defined as:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
where $\hat{y}$ is the predicted value, y is the original value, and n is the number of rows in the test data.
Submission File
For each row in the test set, you must predict the value of the target as described on the data tab, each on a separate row in the submission file. The file should contain a header and have the following format:

    id,target
    0,0.5
    5,10.2
    15,2.2
    etc.


##  Dataset Description:
For this competition, you will be predicting a continuous target based on a number of feature columns given in the data. All of the feature columns, cat0 - cat9 are categorical, and the feature columns cont0 - cont13 are continuous.

Files

    train.csv - the training data with the target column
    test.csv - the test set; you will be predicting the target for each row in this file
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, cat0, cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cont0, cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9, cont10, cont11, cont12, cont13, target
test.csv - column name: id, cat0, cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cont0, cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9, cont10, cont11, cont12, cont13


## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-feb-2021. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Thank you for reading this notebook. I'm new to Kaggle and machine-learing algorithms, and this competition is the second one for me after TPS-January. I didn't use any special techniques, but used GBDT modules I found common in Kaggle (LightGBM, XGBoost, and CatBoost). In this notebook I wrote down the basic flows I used in this competition. I don't suppose this will interest those who has been familiar with Kaggle, but I would appreciate it if you could read this and give me some advice. I'm also glad if this notebook would help other beginners.

#### Feature Engineering
I did a slight feature-engineering. Histograms of the cont features show multiple components. For instance, the cont1 has 7 discrete peaks as shown below. I thought these characteristics could be used as an additional feature. So, I tried sklearn.mixture.GaussianMixture to devide into several groups

For categorical features, I used label-encoding (sklearn.preprocessing.LabelEncoder).

#### Hyperparameter Tuning
I learned that the hyperparameter tuning is necessary to improve scores. Here is the example for tuning LightGBM by Optuna. I don't really know what parameters to tune and what range to input (I don't even know what each parameter means😥 ). Please let me know if I'm missing the point.

#### Model Training
I found training for different random seeds and averaging them improve PB scores. 

I trained two lgm models using different random seeds. The catboost and xgboost models were trained separately. Finally, the outputs of these four models were integrated


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: