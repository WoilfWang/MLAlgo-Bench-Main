You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Predicting_Red_Hat_Business_Value_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Like most companies, Red Hat is able to gather a great deal of information over time about the behavior of individuals who interact with them. They’re in search of better methods of using this behavioral data to predict which individuals they should approach—and even when and how to approach them.

In this competition, Kagglers are challenged to create a classification algorithm that accurately identifies which customers have the most potential business value for Red Hat based on their characteristics and activities.

With an improved prediction model in place, Red Hat will be able to more efficiently prioritize resources to generate more business and better serve their customers.

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted and the observed outcome.

#### Submission File
For each activity_id in the test set, you must predict a probability for the 'outcome' variable, represented by a number between 0 and 1. The file should contain a header and have the following format:

    activity_id,outcome
    act1_1,0act1_100006,0
    act1_100050,0
    etc.

##  Dataset Description:
This competition uses two separate data files that may be joined together to create a single, unified data table: a people file and an activity file.

The people file contains all of the unique people (and the corresponding characteristics) that have performed activities over time. Each row in the people file represents a unique person. Each person has a unique people_id.

The activity file contains all of the unique activities (and the corresponding activity characteristics) that each person has performed over time. Each row in the activity file represents a unique activity performed by a person on a certain date. Each activity has a unique activity_id.

The challenge of this competition is to predict the potential business value of a person who has performed a specific activity. The business value outcome is defined by a yes/no field attached to each unique activity in the activity file. The outcome field indicates whether or not each person has completed the outcome within a fixed window of time after each unique activity was performed.

The activity file contains several different categories of activities. Type 1 activities are different from type 2-7 activities because there are more known characteristics associated with type 1 activities (nine in total) than type 2-7 activities (which have only one associated characteristic).

To develop a predictive model with this data, you will likely need to join the files together into a single data set. The two files can be joined together using person_id as the common key. All variables are categorical, with the exception of 'char_38' in the people file, which is a continuous numerical variable.

act_test.csv - column name: people_id, activity_id, date, activity_category, char_1, char_2, char_3, char_4, char_5, char_6, char_7, char_8, char_9, char_10
people.csv - column name: people_id, char_1, group_1, char_2, date, char_3, char_4, char_5, char_6, char_7, char_8, char_9, char_10, char_11, char_12, char_13, char_14, char_15, char_16, char_17, char_18, char_19, char_20, char_21, char_22, char_23, char_24, char_25, char_26, char_27, char_28, char_29, char_30, char_31, char_32, char_33, char_34, char_35, char_36, char_37, char_38
act_train.csv - column name: people_id, activity_id, date, activity_category, char_1, char_2, char_3, char_4, char_5, char_6, char_7, char_8, char_9, char_10, outcome


## Dataset folder Location: 
../../kaggle-data/predicting-red-hat-business-value. In this folder, there are the following files you can use: sample_submission.csv, act_test.csv, people.csv, act_train.csv

## Solution Description:
Hello guys,
I‘m going to write a story how I did it, because knowing why I did it is hard to tell in few words:)

This competition started as a goal to reach top10 for grandmaster title, which ended up with a stressful race for top1 finalle. Luckily, I made some good decisions in the last day which guaranteed me top1 spot. For me this competition was not something I enjoyed very much but still glad everything worked out in the end. My final model is quite simple and if no leak was present, it might have been production friendly.

First, proper cross-validation set was very important. Tricks I did to have a representative CV set:
1)    Remove observations of group_1=17304, both from train set and test set; That correspons to 30% of training data set, and this group has all outcome=0 (which was used an override rule in making prediction files)
2)    Use Distinct operator for gruop_1‘s which have 3000+ number of rows (this step was very important to remove potential auc bias for several of my CV folds)
3)    Create random unstratified 5-fold cv set based on people file 


My modeling concept was rather simple – reduce original problem to few smaller ones, and combine them in 2nd level model. I have built several models using principles below:

    a)    Select an activity in each group_1‘s timeline (I used first/last activity in a timeline)
    b)    Collect all other activities within a group which has the same outcome label
    c)    Aggregate features – tf-idf was especially useful (in short, for each varriables‘ attribute calculation was done: #of people with same attribute in a group/#of people with same attribute in population)
    d)    Add other simple and not so simple features (i.e. group_1 id value, #activities in a group, #people in a group, min/max dates etc.); I did not use any feature interactions or likelihood features.
    e)    Build a classifier on the dataset – I only used xgboost; was able to reach ~0.84 AUC (bare in mind, no leakage has been used till this point yet!) which in my mind was a fantastic result; If I was RedHat, I would have made this as a target to model, but oh, nevermind.

To do it properly I had to think of some novel cross-validation approach as my split was based on people_id, and on this aggregated data level my CV had to be based on some kind of aggregated CV split scheme as well; My approach worked well, but along the way I had to build ~15 xgboost models to make it work (this was necessary to make 2nd level model work). Such proper CV scheme was important, but I‘m not going into details how it works for now. 

So at this point I have 4 very well performing 1st level models (2 models which perform very well on public LB, and 2 simillar versions which yield best CV score in 2nd level model); 2nd layer model required very careful scripting skills, as I implemented leakge solution in a cross-validated way, so the predictions of outcome changes within group would be learned in a ML way (simple rules in public scripts are not that good!); So in fact 2nd level model solves 2 problems – predict probabilities imputing observations affected by leakage, and predict probabilities for observations not affected by leakage. The model itself is simple but some smart features were included as well to capture time trends within group/population.

So at this point I have very decent model, and in the middle of competition I see myself at #4 place, and I see top3 guys improving their score day by day. So I put some time into it to think about it & discover that they have been doing some manual exploitation using public LB to increase the leakage. As public/private split was random (took some time to discover), one can use a hand-crafted submissions to test groups not affected by leakage to get auc results for that specific gruop, and determine what the probable outcome of whole group is. Do this for as many submissions taking largest group_1‘s as you can, and you might detect some gruops, that ML model misclassifies badly; having that in mind manually create some overrides based on that; So in my submissions i started using rule-based overrides to corret ML model shortcomings.

For my final submission i used a simple average of my best performing LB model and best performing model on CV – this was my last effort to overtake Victor, and for my surprise it provided very high uplift to my score;  I want to thank Victor for not making this easy for me in last few days, as I got myself a little too relaxed at one point:)
p.s. if I did not used any overrides I think I still would have finished within top 5. Sorry for NoHat team, which seems to have made the best ML model there..


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: