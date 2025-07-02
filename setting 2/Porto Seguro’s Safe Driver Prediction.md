You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Porto_Seguro’s_Safe_Driver_Prediction_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Nothing ruins the thrill of buying a brand new car more quickly than seeing your new insurance bill. The sting’s even more painful when you know you’re a good driver. It doesn’t seem fair that you have to pay so much if you’ve been cautious on the road for years.

Porto Seguro, one of Brazil’s largest auto and homeowner insurance companies, completely agrees. Inaccuracies in car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones.

In this competition, you’re challenged to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. While Porto Seguro has used machine learning for the past 20 years, they’re looking to Kaggle’s machine learning community to explore new, more powerful methods. A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more drivers.

##  Evaluation Metric:
#### Scoring Metric
Submissions are evaluated using the Normalized Gini Coefficient.

During scoring, observations are sorted from the largest to the smallest predictions. Predictions are only used for ordering observations; therefore, the relative magnitude of the predictions are not used during scoring. The scoring algorithm then compares the cumulative proportion of positive class observations to a theoretical uniform proportion.

The Gini Coefficient ranges from approximately 0 for random guessing, to approximately 0.5 for a perfect score. The theoretical maximum for the discrete calculation is (1 - frac_pos) / 2.

The Normalized Gini Coefficient adjusts the score by the theoretical maximum so that the maximum score is 1.

The code to calculate Normalized Gini Coefficient in a number of different languages can be found in this forum thread.

#### Submission File
For each id in the test set, you must predict a probability of an insurance claim in the target column. The file should contain a header and have the following format:

    id,target
    0,0.1
    1,0.9
    2,1.0
    etc.

##  Dataset Description:
#### Data Description
In this competition, you will predict the probability that an auto insurance policy holder files a claim.

In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.
#### File descriptions

    train.csv contains the training data, where each row corresponds to a policy holder, and the target columns signifies that a claim was filed.
    test.csv contains the test data.
    sample_submission.csv is submission file showing the correct format.

train.csv - column names: id, target, and other feature names. 
test.csv - column names: id, and other feature names. 

## Dataset folder Location: 
../../kaggle-data/porto-seguro-safe-driver-prediction. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:

My solution is not very complex, just an average of 1 lgb and 1 nn, built pretty much on the same feature space. Most important parts:

Feature elimination. I dropped all of calc features and ['ps_ind_14','ps_car_10_cat','ps_car_14','ps_ind_10_bin','ps_ind_11_bin',
'ps_ind_12_bin','ps_ind_13_bin','ps_car_11','ps_car_12']. I was excluding them one by one in greedy fashion and checking lgb cross validation score.
Hot encoding categorical variables. It helped to reduce noise while getting the splits for most useful categories.
For NN model it was also necessary to hot encode numeric features with small number of unique values - ['ps_car_15','ps_ind_01','ps_ind_03','ps_ind_15','ps_reg_01','ps_reg_02'] (without dropping the original ones)
Regularized models. lgb_par = {'feature_fraction': 0.9, 'min_data_in_leaf': 24, 'lambda_l1':10, 'bagging_fraction': 0.5, 'learning_rate': 0.01, 'num_leaves': 24}
Another thing that I unfortunately haven't explored well is anomaly detection on train+test datasets. Just like less frequent categories (or combinations) of categories are more likely to have label 1, we could find 'strange' samples via unsupervised methods. For example, if we train a basic autoencoder, AUC score of sample-wise reconstruction error would be ~0.60, which is pretty high. I believe more thorough analysis could make this approach really useful.

Generally, it's hard to tell what else did not really work or could have worked: almost everything that you will try to do in a competition like this will result in no significant change, no matter whether you did it right or wrong:)

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: