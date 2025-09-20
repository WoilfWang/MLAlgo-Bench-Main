You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Binary_Classification_with_a_Tabular_Pulsar_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the 2023 edition of Kaggle's Playground Series!

Thank you to everyone who participated in and contributed to Season 3 Playground Series so far! 

With the same goal to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science, we will continue launching the Tabular Tuesday in March every Tuesday 00:00 UTC, with each competition running for 2 weeks. Again, these will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.. 

### Synthetically-Generated Datasets
Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

##  Evaluation Metric:
Submissions are scored on the log loss:
$$
\textrm{LogLoss} = - \frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right],
$$
where

    \( n \) is the number of rows in the test set
    \( \hat{y}_i \) is the predicted probability the Class is a pulsar
    \( y_i  \) is 1 if Class is pulsar, otherwise 0
    \( log \) is the natural logarithm

The use of the logarithm provides extreme punishments for being both confident and wrong. In the worst possible case, a prediction that something is true when it is actually false will add an infinite amount to your error score. In order to prevent this, predictions are bounded away from the extremes by a small value.

#### Submission File
For each id in the test set, you must predict the value for the target Class. The file should contain a header and have the following format:

    id,Strength
    117564,0.11
    117565,0.32
    117566,0.95
    etc.


##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Pulsar Classification. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Files

    train.csv - the training dataset; Class is the (binary) target
    test.csv - the test dataset; your objective is to predict the probability of Class (whether the observation is a pulsar)
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, Mean_Integrated, SD, EK, Skewness, Mean_DMSNR_Curve, SD_DMSNR_Curve, EK_DMSNR_Curve, Skewness_DMSNR_Curve, Class
test.csv - column name: id, Mean_Integrated, SD, EK, Skewness, Mean_DMSNR_Curve, SD_DMSNR_Curve, EK_DMSNR_Curve, Skewness_DMSNR_Curve


## Dataset folder Location: 
../../kaggle-data/playground-series-s3e10. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:

3rd solution is a blend of XGBoost, LightGBM and Generalized Additive Models here :

    with from 4 to 10 folds Stratified CV

    with log transformation of some features

    with some interactions features

    with forward selection of interactions features and values permutations of predictions to do backward features selection and to drop useless features and to avoid overfitting.

    with weak learners for XGBoost and LightGBM : only 5 ou 6 leaves

    with overfitting control by computing difference between val_logloss and trn_logloss for each fold, to fit regularization hyperparameters in XGBoost and LightGBM, with the hands (See @ambrosm in https://www.kaggle.com/competitions/playground-series-s3e9/discussion/394592)

    hence without optuna or other optimization tools

    without calibration of predictions, because by CV I saw it was useless

    without original data.

    with a little mistake in final submission in diversity of models, which costs me the second place (see difference betwwen version v40 & v39 - v40 was what i wanted to do but I did v39).

By computing difference between val_logloss and trn_logloss, we can see that there was less over-fitting with GAM than with GBMs.

I tried RandomForest and ExtraTrees, but my CV was only 0.033.

 tried to fit a LogisticRegression without sucess, I wasn't able to add interactions with polynomialfeatures, thank's @ambrosm for your solution, I'll read it carefully.

Special thank's to @paddykb for GAM in this notebook and log transformation of some features.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: