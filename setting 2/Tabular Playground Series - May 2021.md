You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_May_2021_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly. 

In order to have a more consistent offering of these competitions for our community, we're trying a new experiment in 2021. We'll be launching month-long tabular Playground competitions on the 1st of every month and continue the experiment as long as there's sufficient interest and participation.

The goal of these competitions is to provide a fun, and approachable for anyone, tabular dataset. These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you. We encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams, to inspire broad participation we are limiting winner's of swag to once per person for this series. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals. 

The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting the category on an eCommerce product given various attributes about the listing. Although the features are anonymized, they have properties relating to real-world features.


##  Evaluation Metric:
Submissions are evaluated using multi-class logarithmic loss. Each row in the dataset has been labeled with one true Class. For each row, you must submit the predicted probabilities that the product belongs to each class label. The formula is:

$$ \text{log loss} = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}\log(p_{ij}), $$
where \(N\) is the number of rows in the test set, \(M\) is the number of class labels, \( \text{log}\) is the natural logarithm, \(y_{ij}\) is 1 if observation \(i\) is in class \(j\) and 0 otherwise, and \(p_{ij}\) is the predicted probability that observation \(i\) belongs to class \(j\).

The submitted probabilities for a given product are not required to sum to one; they are rescaled prior to being scored, each row is divided by the row sum. In order to avoid the extremes of the \(\text{log}\) function, predicted probabilities are replaced with \(max(min(p,1-10^{-15}),10^{-15})\).

Submission File

You must submit a csv file with the product id and the predicted probability that the product belongs to each of the classes seen in the dataset. The order of the rows does not matter. The file must have a header and should look like the following:

    id,Class_1,Class_2,Class_3,Class_4
    100000,0.1,0.3,0.2,0.4
    100001,0.5,0.1,0.1,0.3
    100002,0.4,0.4,0.1,0.1
    etc.


##  Dataset Description:
The dataset is used for this competition is synthetic, but based on a real dataset and generated using a CTGAN. The original dataset deals with predicting the category on an eCommerce product given various attributes about the listing. Although the features are anonymized, they have properties relating to real-world features.

Files

    train.csv - the training data, one product (id) per row, with the associated features (feature_*) and class label (target)
    test.csv - the test data; you must predict the probability the id belongs to each class
    sample_submission.csv - a sample submission file in the correct format

train.csv - column names: id, target, feature_0, feature_1, ..., feature_49.
test.csv - column names: id, feature_0, feature_1, ..., feature_49.

## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-may-2021. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Hey folks, 

This is my first post on kaggle after 4 years. I never really participated since signing up until this year with the introduction of the Tabular playground series. These competitions prompted me to participate more and I have been having a lot of fun with these and learning a lot from the amazing people in this community sharing their work. I just personally need to get better at sharing and voting, so here is a start.

I am a bit surprised by this result as my solution is really boring and nothing fancy. I also am not convinced that this solution is a good classifier as it still is struggling with rare classes (ensembling across random seeds reduced that effect though.). I think what worked at the end of was trusting my local cv results and not using public lb and not blindly using the automl results. Don't get me wrong those are very helpful but in this competition they did not help my local cv so I did not use them at the end.

My workflow is in Google Colab and my code is quite messy and at time disorganized. With a very hectic day job, I did not have much time to clean things up but still thought to share in case someone finds any of this useful. Below is also a description of what worked at the end. 

Here is the bulk of my code on Google Colab for individual models and cv (does not include the stacking stage). 

#### Feature Selection/Engineering
Nothing worked for me here. I tried PCA, clustering, also played with featuretools but all led to either horrible overfitting or poor local cv so at the end I did not pursue feature engineering much. 

For feature selection, due to the non existence of any dominant features or highly associated features with the target, I suspected that removing features would not be helping the cv so I kept all the features. 

For the categorical features, for my linear model (Logistic Regression) I used onehot encoding using scipy sparse features to fit in-memory, and for all the other models (all tree based) I just used OrdinalEncoder (essentially label encoding). 

#### Base Models
I ended up training 5 different models, each trained on a 5fold stratified cv on three random seeds, using out of fold predictions for each fold and averaging across folds for predicting the test set. The models were LogisticsRegression, RandomForest, XGBoost, LightGBM, and CATBoost. Weighting classes did not really work for LogisticsRegression and I used the defaults class weights in the rest of them.  I used Optuna to lightly tune each model (you will see my final parameters in the code above). 

#### Over/Under Sampling
Did not really try after reading the notebook from @remekkinas that pointed out it likely would not work. His analysis made sense to me and I did not have time, so did not really pursue any further.
#### Ensembling
Stacking worked quite well here. I tried weighted average of models and it did worse than stacking so I did not use that. I used a 5fold stacking of all models above and used a meta model of RidgeRegerssion, using CalibratedClassifierCV in sklearn. 
#### Final submission
I ended up clipping the probabilities (below 0.05 and above .95) to help the log_loss metric. This  article explains very briefly (clipping helps mitigating the extremes of too small or too large values in the log_loss metric). Final submission had a public LB of 1.08564 and private of 1.08763. Looks like blending with top public notebooks could have brought the private LB scoe down to 1.08742 which I did not do.



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: