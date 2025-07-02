You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Categorical_Feature_Encoding_Challenge_II_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Can you find more cat in your dat?
We loved the participation and engagement with the first Cat in the Dat competition.
Because this is such a common task and important skill to master, we've put together a dataset that contains only categorical features, and includes:

    binary features
    low- and high-cardinality nominal features
    low- and high-cardinality  ordinal features
    (potentially) cyclical features

This follow-up competition offers an even more challenging dataset so that you can continue to build your skills with the common machine learning task of encoding categorical variables.  This challenge adds the additional complexity of feature interactions, as well as missing data.

This Playground competition will give you the opportunity to try different encoding schemes for different algorithms to compare how they perform. We encourage you to share what you find with the community.

If you're not sure how to get started, you can check out the Categorical Variables  section of Kaggle's Intermediate Machine Learning course.

Have Fun!

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

Submission File

For each id in the test set, you must predict a probability for the target variable. The file should contain a header and have the following format:

    id,target
    600000,0.5
    600001,0.5
    600002,0.5
    ...


##  Dataset Description:
In this competition, you will be predicting the probability [0, 1] of a binary target column.

The data contains binary features (bin_*), nominal features (nom_*), ordinal features (ord_*) as well as (potentially cyclical) day (of the week) and month features. The string ordinal features ord_{3-5} are lexically ordered according to string.ascii_letters.

Since the purpose of this competition is to explore various encoding strategies. Unlike the first Categorical Feature Encoding Challenge, the data for this challenge has missing values and feature interactions.

Files

    train.csv - the training set
    test.csv - the test set; you must make predictions against this data
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, bin_0, bin_1, bin_2, bin_3, bin_4, nom_0, nom_1, nom_2, nom_3, nom_4, nom_5, nom_6, nom_7, nom_8, nom_9, ord_0, ord_1, ord_2, ord_3, ord_4, ord_5, day, month, target
test.csv - column name: id, bin_0, bin_1, bin_2, bin_3, bin_4, nom_0, nom_1, nom_2, nom_3, nom_4, nom_5, nom_6, nom_7, nom_8, nom_9, ord_0, ord_1, ord_2, ord_3, ord_4, ord_5, day, month


## Dataset folder Location: 
../../kaggle-data/cat-in-the-dat-ii. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Congrats to all the participants in this great and challenging tabular competition! Thank you  Kaggle for organizing this competition.

My solution is very simple, NN plays a key role. The categorical features, especially with high cardinality are very suitable for neural network to exert its power.

#### About Models
4x NN,  1x Catboost, and blend of some of public kernels.  All NN models on same features,  and one of them resulted in 0.78672 on the public LB. Thanks to @sergey and @siavash for sharing the public kernels, I used them in blending.
Feature engineering for NN models

    All features are converted into category type then Label Encoding
    Ordinal Encoding: ord_1~ord_5 -> ord_1_en~ord_5_en

That’s all. I have also tried using various other methods to process the features but they all lead to overfitting.

NN Models

NN uses several state-of-the-art models for CTR prediction, including CIN in xDeepFM, PNN, Cross in DCN, AutoInt, etc. You can use the libray of deeptables to implement these models.

    1.Linear+DNN+CIN (0.78672 public LB)
    2.FM+Cross+PNN (0.78655 public LB)
    3.FM+DCN+DNN (0.78652 public LB)
    4.Linear+DNN+AutoInt (0.78665 public LB)
    
There are many components available for feature extraction on tabular data and they can be combined with various ways. It is a huge workload to trail by coding from scratch every time. Deeptables greatly simplifies this job with only a few lines of code.



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: