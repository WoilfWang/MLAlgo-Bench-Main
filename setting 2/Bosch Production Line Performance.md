You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Bosch_Production_Line_Performance_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
A good chocolate soufflé is decadent, delicious, and delicate. But, it's a challenge to prepare. When you pull a disappointingly deflated dessert out of the oven, you instinctively retrace your steps to identify at what point you went wrong. Bosch, one of the world's leading manufacturing companies, has an imperative to ensure that the recipes for the production of its advanced mechanical components are of the highest quality and safety standards. Part of doing so is closely monitoring its parts as they progress through the manufacturing processes.

Because Bosch records data at every step along its assembly lines, they have the ability to apply advanced analytics to improve these manufacturing processes. However, the intricacies of the data and complexities of the production line pose problems for current methods.

In this competition, Bosch is challenging Kagglers to predict internal failures using thousands of measurements and tests made for each component along the assembly line. This would enable Bosch to bring quality products at lower costs to the end user.

##  Evaluation Metric:
Submissions are evaluated on the Matthews correlation coefficient (MCC) between the predicted and the observed response. The MCC is given by:
$$MCC = \frac{(TP*TN) - (FP * FN)}{\sqrt{(TP+FP)(TP+FN)(TN + FP)(TN+FN)}},$$
where TP is the number of true positives, TN the number of true negatives, FP the number of false positives, and FN the number of false negatives.

#### Submission File
For each Id in the test set, you must predict a binary prediction for the Response variable. The file should contain a header and have the following format:

    Id,Response
    1,0
    2,1
    3,0
    etc.

##  Dataset Description:
The data for this competition represents measurements of parts as they move through Bosch's production lines. Each part has a unique Id. The goal is to predict which parts will fail quality control (represented by a 'Response' = 1).

The dataset contains an extremely large number of anonymized features. Features are named according to a convention that tells you the production line, the station on the line, and a feature number. E.g. L3_S36_F3939 is a feature measured on line 3, station 36, and is feature number 3939.

On account of the large size of the dataset, we have separated the files by the type of feature they contain: numerical, categorical, and finally, a file with date features. The date features provide a timestamp for when each measurement was taken. Each date column ends in a number that corresponds to the previous feature number. E.g. the value of L0_S0_D1 is the time at which L0_S0_F0 was taken.

In addition to being one of the largest datasets (in terms of number of features) ever hosted on Kaggle, the ground truth for this competition is highly imbalanced. Together, these two attributes are expected to make this a challenging problem.

### File descriptions

    train_numeric.csv - the training set numeric features (this file contains the 'Response' variable)
    test_numeric.csv - the test set numeric features (you must predict the 'Response' for these Ids)
    train_categorical.csv - the training set categorical features
    test_categorical.csv - the test set categorical features
    train_date.csv - the training set date features
    test_date.csv - the test set date features
    sample_submission.csv - a sample submission file in the correct format

In these csv files, the first column name of all files is Id. 

## Dataset folder Location: 
../../kaggle-data/bosch-production-line-performance. In this folder, there are the following files you can use: train_categorical.csv, sample_submission.csv, test_date.csv, train_numeric.csv, test_categorical.csv, test_numeric.csv, train_date.csv

## Solution Description:
Sorry for the delay. It was good to have a kaggle free weekend after two months. (Ok not completely kaggle free but at least in read only mode:).

As we had 6-7 hours local time difference (USA, EU) we tried to split the work. 

I was mostly doing feature engineering and tried to improve our best single xgb score. Ash tried all kinds of different modeling techniques and built the L2/L3 ensemble.
### Data Exploration
I spent the first two weeks with data discovery. Numeric feature plots, statistics, station to station transition probabilities, correlation matrices for each stations was necessary to have ideas how to handle the thousands of anonymous features.

I start each competition with desktop machine (16GB RAM). It takes some extra effort in the beginning to deal with lower level data manipulations due to the memory constraints. Getting closer to the raw data helps sometimes later. 

As the numeric/date features had .3f/.2f precision we could rescale (*=1000, *=100) everything to integers. We kept only the min time for each station.

Sparse matrices came handy as we had a lot of missing values. 
### Leak/Magic/Data Property Features
Ash found quite early that consecutive rows had feature duplication and correlated Response. We called consecutive Ids ordered by StartStation, StartTime chunks.

For each chunk we used features

    number of records
    rank ascending
    rank descending

We used the original order to add features based on previous and next records.

    Response
    feature hash equality for each station
    StartTime, EndTime
    Numeric raw features

### Time Features
I figured out in the beginning that 0.01 time granularity means probably 6 mins.

So we had to deal with 2 years of manufacturing data. This observation did not gave direct performance boost but gave us enough intuition to construct a lot of time based features.

    StartStationTimes
    StartTime, EndTime, Duration
    StationTimeDiff
    Start/End part of week (mod 1680)
    Number of records in next/last 2.5h, 24h, 168h for each station
    Number of records in the same time (6 mins)
    MeanTimeDiff since last 1/5/10 failure(s)
    MeanTimeDiff till next 1/5/10 failure(s)

### Numeric Features

    Raw numeric features (most of the time we used the raw numeric features or simple subsets based on xgb feature importance)
    Z-scaled features for each week
    Count encoding for each value 
    Feature combinations (f1 + -  * f2)

We saw interesting similarity in the station transition probability matrix between station S0-S11 and S12 - S23. You have probably seen the same in the Shopfloor visualizations. We also noticed that the number of features are the same across these station groups.

The correlation matrices were similar for these stations so we tried combining the numeric features for the same stages. (e.g. L0_S0_F0 + L0_S12_F330 etc.)

### Categorical Features
We could not squeeze much out of the categorical features. Most of the time we just dropped them. We had a few models with aggregated categorical features for each station in [24, 25, 26, 27, 28, 29, 32, 44, 47]. 'T1' was replaced by 1 all the other values were replaced 100.
### Validation
Given the imbalanced labels and binary evaluation criteria we noticed quite high 5-FOLD CV stds (~0.01) in the beginning. After a few experiments we decided to use 4 FOLD "leak-stratified" CV (forcing to have the same number of duplication for each fold). 

Each "single" L1 model was trained with 3 different seeds on the same folds. Our latest ensemble scores were quite stable (std < 0.004) and very often our CV/LB score improved hand in hand.
### Rebalancing
We kept every failure and records with duplication. For the remaining 0s 50-90% down sampling was used. It made the training a bit faster and worked as bagging for the model averaging part helping the later ensemble stages.
### Results
In the last month of the competition we continuously improved and tried to keep us in the top 3. Fortunately we could improve significantly both CV and LB  with our last submissions on Friday. This made the submission selection easier.
imagehttps://kaggle2.blob.core.windows.net/forum-message-attachments/144544/5348/results.PNG?sv=2012-02-12&se=2016-11-17T16%3A01%3A24Z&sr=b&sp=r&sig=%2FBvQoJg5kEuzvBWFBoHFs7GU%2Fgr7AotglS54yYgZk3M%3D.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: