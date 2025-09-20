You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Santander_Customer_Transaction_Prediction_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
At Santander  our mission is to help people and businesses prosper.  We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals. 

Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure  we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?

In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

Submission File

For each Id in the test set, you must make a binary prediction of the target variable. The file should contain a header and have the following format:

    ID_code,target
    test_0,0
    test_1,1
    test_2,0
    etc.

##  Dataset Description:
You are provided with an anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code column.

The task is to predict the value of target column in the test set.

File descriptions

    train.csv - the training set.
    test.csv - the test set. The test set contains some rows which are not included in scoring.
    sample_submission.csv - a sample submission file in the correct format.

train.csv - column names: ID_code, target, var_0, var_1, ..., var_199. 
test.csv - column names: ID_code, var_0, var_1, ..., var_199. 


## Dataset folder Location: 
../../kaggle-data/santander-customer-transaction-prediction. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
First, congratulations to every team that participated and fought hard to find this so called "magic", and especially to The Zoo for smoothing the way to .92x and all the top teams that scared us up to the last minute … !

This solution write-up will contain technical parts as well as, as many persons ask, some details about my journey that led to our solution. Feel free to read what interests you :)

TLDR:
made 400 features,
LGBM, 600feats, that scores 0.92522 public /0.92332 private using pseudo-label and data augmentation
Winning ticket : NN, 600feats, with custom structure that scores .92687 public / 0.92546 private using pseudo-label and data augmentation
blending them (2.1NN, 1LGBM) gave us our .927 public

#### Feature engineering:
Technical part:
The "magic" is about count of values, especially the fact that some are unique.
We created 200 (one per raw feature) categorical features, let's call them "has one feat", with 5 categories that corresponds (for train data) to: 

    This value appears at least another time in data with target==1 and no 0; 
    This value appears at least another time in data with target==0 and no 1;
    This value appears at least two more time in data with target==0 & 1; 
    This value is unique in data;
    This value is unique in data + test (only including real test samples);

The other 200 (one per raw feature) features are numerical, let's call them "not unique feat", and correspond to the raw feature replacing values that are unique in data + test with the mean of the feature.

#### My journey to the findings:
After some EDA where key insight was that number of different values in train and test was not the same, I started with LGBM, because it is fast and powerful and easy to use.

Like many I began to see CV/LB improvements with count encoding of features.

I looked at my LGBM trees (with only 3 leafs that's easy to do) and noticed the trees were using the uniqueness information.

After this insight, I started to build features around uniqueness. Using only training data and the "has one feat", I could reach .910 LB. Adding the other 200 "not unique feat", .914LB. 

The next move was to use data + test to spot unique values. It worked really well on CV, giving >.92x results but didn't apply to test as is! 

As many people noticed, the count of unique values per feature in data and test is very different! So I knew that there was a subset of samples in test that I couldn't identify yet that would bring >.92x LB. I teamed with Silogram at this moment. The day after he sent me a link to the beautiful and very important kernel of @YaG320 (rick and morty's fans are the best!) "List of Fake Samples and Public/Private LB split". I immediately understood that this was the key to spot values that are unique in data + test!

We got LB .921 using LGBM at this time, and these are the features we used at the end.

#### Modelisation:
Technical part:

We used standard 10 fold Stratified cross validation with multiple seeds for final blend.

We made a LGBM using the shuffle augmentation (duplicate and shuffle 16 times samples with target == 1, 4 for target ==0) and added pseudo label (2700 highest predicted test points as 1 and 2000 lowest as 0). Our LGBM performs .92522 Public, .92332 private.

Our second model was a NN with a particular structure: 

The idea, like many did, was to process all the features belonging to the same group(raw / has one / not unique) independently and in the same way (i.e using same set of weights). That would create sort of embedding of this feature value. What differentiate us is the next step : We did a weighted average of those 200 embeddings which we then feed to a dense layer for final output. This ensure that every feature is treated in the same way. The weights were generated by another NN. The idea is very similar to what attention networks do. Everything was of course optimized end to end.
We added on the fly augmentation (for every batch, shuffle the features values that belong to target == 1 / target == 0) and it scored .92497 private. Adding pseudo label (5000 highest and 3000 lowest) increased private to .92546.
Our final submission is a blend of these 2 models with weight 2.1 NN / 1 LGBM.


I really recommend it to everyone … For the training of the neural network It also made things easy : I added batch norm and small dropouts almost everywhere and then the fit one cycle method with 15 epochs at 0.01 learning rate (nothing fancy) was enough to achieve those results!


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: