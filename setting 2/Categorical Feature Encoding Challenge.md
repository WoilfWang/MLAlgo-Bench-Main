You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Categorical_Feature_Encoding_Challenge_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Is there a cat in your dat?

A common task in machine learning pipelines is encoding categorical variables for a given algorithm in a format that allows as much useful signal as possible to be captured.

Because this is such a common task and important skill to master, we've put together a dataset that contains only categorical features, and includes:

    binary features
    low- and high-cardinality nominal features
    low- and high-cardinality  ordinal features
    (potentially) cyclical features

This Playground competition will give you the opportunity to try different encoding schemes for different algorithms to compare how they perform. We encourage you to share what you find with the community.

If you're not sure how to get started, you can check out the Categorical Variables  section of Kaggle's Intermediate Machine Learning course.
Have Fun!

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

Submission File

For each id in the test set, you must predict a probability for the target variable. The file should contain a header and have the following format:

    id,target
    300000,0.5
    300001,0.5
    300002,0.5
    ...


##  Dataset Description:
In this competition, you will be predicting the probability [0, 1] of a binary target column.

The data contains binary features (bin_*), nominal features (nom_*), ordinal features (ord_*) as well as (potentially cyclical) day (of the week) and month features. The string ordinal features ord_{3-5} are lexically ordered according to string.ascii_letters.

Since the purpose of this competition is to explore various encoding strategies, the data has been simplified in that (1) there are no missing values, and (2) the test set does not contain any unseen feature values (See this). (Of course, in real-world settings both of these factors are often important to consider!)

Files

    train.csv - the training set
    test.csv - the test set; you must make predictions against this data
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, bin_0, bin_1, bin_2, bin_3, bin_4, nom_0, nom_1, nom_2, nom_3, nom_4, nom_5, nom_6, nom_7, nom_8, nom_9, ord_0, ord_1, ord_2, ord_3, ord_4, ord_5, day, month, target
test.csv - column name: id, bin_0, bin_1, bin_2, bin_3, bin_4, nom_0, nom_1, nom_2, nom_3, nom_4, nom_5, nom_6, nom_7, nom_8, nom_9, ord_0, ord_1, ord_2, ord_3, ord_4, ord_5, day, month


## Dataset folder Location: 
../../kaggle-data/cat-in-the-dat. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
I thought about whether to post my solution for a long time because it may not be better than other complicated solutions. 

But it is really simple and performs well. It’s more like a baseline model. I hope you can learn something from this. : )
What I did:

    Dropping bin_0
    Ordinal Encoding ord features
    One Hot Encoding other features
    Using logistic regression with ‘liblinear’ solver
    Tuning C by using optuna

I did these things except optuna at the beginning of this game.

I tuned C the last day and trained model all at once instead of using Kfolds. It improved my score to 0.80850
I’m new to machine learning and this is my first competition. 

Thanks to Kaggle and everyone participates in this competition. I learned a lot from you guys, which is more important than the ranking. Thanks for your share!.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: