You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_Jan_2022_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
We've heard your feedback from the 2021 Tabular Playground Series, and now Kaggle needs your help going forward in 2022!
There are two (fictitious) independent store chains selling Kaggle merchandise that want to become the official outlet for all things Kaggle. We've decided to see if the Kaggle community could help us figure out which of the store chains would have the best sales going forward. So, we've collected some data and are asking you to build forecasting models to help us decide. 

Help us figure out whether KaggleMart or KaggleRama should become the official Kaggle outlet!

About the Tabular Playground Series

Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly.

The goal of these competitions is to provide a fun and approachable-for-anyone tabular dataset to model. These competitions are a great choice for people looking for something in between the Titanic Getting Started competition and the Featured competitions. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you; thus, we encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals.

##  Evaluation Metric:
Submissions are evaluated on SMAPE between forecasts and actual values. We define SMAPE = 0 when the actual and predicted values are both 0.

Submission File

For each row_id in the test set, you must predict the corresponding num_sold. The file should contain a header and have the following format:

    row_id,num_sold
    26298,100
    26299,100
    26300,100
    etc.


##  Dataset Description:
For this challenge, you will be predicting a full year worth of sales for three items at two stores located in three different countries.  This dataset is completely fictional, but contains many effects you see in real-world data, e.g., weekend and holiday effect, seasonality, etc. The dataset is small enough to allow you to try numerous different modeling approaches. 

Files

    train.csv - the training set, which includes the sales data for each date-country-store-item combination. 
    test.csv - the test set; your task is to predict the corresponding item sales for each date-country-store-item combination. Note the Public leaderboard is scored on the first quarter of the test year, and the Private on the remaining.
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: row_id, date, country, store, product, num_sold
test.csv - column name: row_id, date, country, store, product


## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-jan-2022. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
The following lines describe the development of my final submission to this competition.
### Importance of cross-validation
In this January TPS one could practice ignoring the public leaderboard. The public leaderboard is based on the first quarter of 2019, but all the interesting holidays occur in April of 2019 or later. This means that the public leaderboard gives no information at all about the quality of a model's holiday features. You can over- or underestimate the influence of Easter, Midsummer Day, National Day, Christmas and so on - for the public leaderboard it doesn't matter. The public leaderboard is only good to verify whether the model deals correctly with the yearly GDP.

For this reason, I focused on cross-validation (GroupKFold with the years as groups), and in the cross-validation results, I evaluated the SMAPE for January through March separately from SMAPE for the rest of the year. Then I consistently optimized my model for the latter. For the final evaluation, I submitted the two notebooks with the best cv. The winning notebook has a public lb score of only 4.11991, which would rank it at position 306 of the public lb. It took quite some courage to mark this as the final submission…

#### Feature engineering
My final notebook still uses Ridge regression with a log-transformed target, but the features differ from my earlier linear model:

    The selection of Fourier coefficients has changed; the stickers get no Fourier coefficients at all (this means that the prediction for the stickers is constant over the whole year).
    There are small changes in the length of holidays.
    The Easter holiday in Norway differs from the Easter holiday in the other two countries.
    I added the OECD's consumer confidence index as external data, as suggested in this discussion.

All these features were found by a detailed analysis of the residuals.

#### Why not gradient boosting?
Why didn't I use gradient boosting? The main advantage of gradient boosting in this competition is that it reduces the burden of feature engineering: Decision trees determine automatically which countries and products a holiday affects; linear regression needs manually crafted features. 

The disadvantage is: If you use gradient boosting and don't engineer the features yourself, you give up control. With linear regression you analyze residuals and create an Easter holiday which starts exactly on Good Friday and lasts ten days, with gradient boosting you tune some hyperparameters and accept that the decision trees may find the holiday to last nine or eleven days. If you tune the hyperparameters so that the holiday has exactly ten days, the model will overfit somewhere else.

#### Varia

    Regularization of ridge regression is controlled by a single parameter, alpha. It is possible to make the regularization strength depend on the feature by scaling features differently: I did not simply use a StandardScaler, but a ColumnTransformer with several MinMaxScalers.
    The ratio between KaggleRama and KaggleMart sales is always the same and does not depend on any other features. A direct calculation is more accurate than linear regression..


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: