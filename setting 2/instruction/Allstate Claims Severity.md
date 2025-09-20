You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Allstate_Claims_Severity_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
When you’ve been devastated by a serious car accident, your focus is on the things that matter the most: family, friends, and other loved ones. Pushing paper with your insurance agent is the last place you want your time or mental energy spent. This is why Allstate, a personal insurer in the United States, is continually seeking fresh ideas to improve their claims service for the over 16 million households they protect.

Allstate is currently developing automated methods of predicting the cost, and hence severity, of claims. In this recruitment challenge, Kagglers are invited to show off their creativity and flex their technical chops by creating an algorithm which accurately predicts claims severity. Aspiring competitors will demonstrate insight into better ways to predict claims severity for the chance to be part of Allstate’s efforts to ensure a worry-free customer experience.
New to Kaggle? This competition is a recruiting competition, your chance to get a foot in the door with the hiring team at Allstate.

##  Evaluation Metric:
Submissions are evaluated on the mean absolute error (MAE) between the predicted loss and the actual loss.

**Submission File**
For every id in the test set, you should predict the loss value. The file should contain a header and have the following format:

    id,loss4,06,19,99.3etc.

##  Dataset Description:
Each row in this dataset represents an insurance claim. You must predict the value for the 'loss' column. Variables prefaced with 'cat' are categorical, while those prefaced with 'cont' are continuous.

**File descriptions**

    train.csv - the training set
    test.csv - the test set. You must predict the loss value for the ids in this file.
    sample_submission.csv - a sample submission file in the correct format

train.csv - column names: id, loss, cat1, cat2, ..., cat116, cont1, cont2, ..., cont14. 
test.csv - column names: id, cat1, cat2, ..., cat116, cont1, cont2, ..., cont14. 

## Dataset folder Location: 
../../kaggle-data/allstate-claims-severity. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:

This is the solution of **Bishwarup**, who ranked 1st in the private leaderboard with the score of **1109.70772**.

Nothing extra-oridinary but a lot of diverse models and data preprocessing. So after my 1st submission which scored 1118 on public LB and I landed the 3rd position at the very beginning of the competiton, I thought I would give a try for my solo gold medal here. I had just relocated to a new city and joined a new company at that moment, but I tried my best to keep in touch with the leaderboard progression and hence update my models going ahead.

### Best Single Model:
My best single model is an XGBoost with the two-way interactions reported on the forum along with some chosen three, four, five and seven-way interactions. It score 1105.12 in the public LB with 10-fold CV around 1124. I chose the higher order interactions primarily based on mutual information. I used fair_obj with a constant of 0.7 for training that model and (1 + loss)**0.25 transformation reported by MrOijer.

### My NN architecture
My best NN scores 1110 in the public leaderboard with a 10-fold cv of 1134.xx. It is very similar to the public scirpt scoring 1111.8x , just some more neurons in each layer.  I used EarlyStopping, ModelCheckpoint and derived metric to monitor (I used log(loss + SHIFT) transformation) and 10-fold 10 times bagged model. Almost all my NN models follow the same architecture.
I used mostly the above two models to train a stacked model (details later in the post) in the second layer, but there are other models with significant contribution. Among them:

#### Regularized Greedy Forest
I had three RGF mixed in to my ensemble trained with slighly different data and loss transformation. My best RGF scores 1113.xx with a cv around 1136.xx. It is a fantastic library that I explored back in BNP Paribas competiton and was determined to use it in one of my competitions. Just one setback is that it's a bit on the slower side being single-threaded.

#### LightGBM
Another very fast and efficient gradient boosting library beside XGboost. I loved it. Best single model 1111.xx on public LB.
Vowpal Wabbit. Always found it worth giving a shot and yes it beat random forest and extra trees which I never could get executed with MAE criterion in sklearn. Best single model 1133.xx with a cv of 1154.xx. I used quadtraic interactions in the categorical namespace.

#### LibFM 
Not much from single model, but definitely helped my ensemble. Best model 1166.xx with 10-fold cv of 1188.xx.

#### LibFFM

Almost at par with my VW. Best model 10-fold cv 1158.xx with public LB 1139.xx. I also used RF(R-h20), ExtraTrees(sklearn), glmnet (R) and some other models to diversify my final ensemble but none of them yielded significant individual performance.

#### Stacking pipeline
I used 10-fold stacking in this competiton and tried a number of preprocessing on the data:

Different order interactions between categorical features.
    
- TF-IDF on categorical features.
- Category Embedding with NN and different embedding layer sizes.
- Standard Scalar/ Minmax scalar
- Different loss function in xgb incuding count:poisson on the rounded and log   transformed targets. Surprisingly it gave me a public LB of 1168.xx.
- Trained XGB on the bottom 5 and 10 percentile and top 70, 80 and 90 percentile of data. The motivation was to get a better estimate of the exorbitant loss values.

I had a total of 81 models in the 1st level and trained an XGB and two NN on them on the second layer.

#### Best Second Level Model
My best L2 model has a 10-fold cv of 1114.645 and public LB 1097.87.

#### Final Submission
My final submission is an wegthed average of the following: 

    w1*NN1^w2 + w3*NN2^w4 + w5*XGB1^w6 + w7 - weights optimized by using optim (Nelder-Mead) in a 1-fold manner => apply weights to test predictions => average 10 test predictions for 10x optimized weights.
    
    If NN1 < w1 , then w2NN1^w3 + w4 Else if  NN1 > w5, then w6NN1^w7 + w8 Else NN1

I extend my congratulations to all other winners and my sincere thanks to Kaggle and Allstate group for arranging this exciting competition. 
Special thanks to Scirpus, Vladimir, Tilli, Lauea (as always), Danijel, d3miekno and all the other people who kept the forum so active all the time with their brilliant ideas and scripts :).


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: