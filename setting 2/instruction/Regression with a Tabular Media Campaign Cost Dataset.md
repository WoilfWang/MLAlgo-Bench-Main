You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Regression_with_a_Tabular_Media_Campaign_Cost_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Thank you to everyone who participated in and contributed to Season 3 Playground Series so far! 

With the same goal to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science, we will continue launching the Tabular Tuesday in March every Tuesday 00:00 UTC, with each competition running for 2 weeks. Again, these will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.. 

Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

##  Evaluation Metric:
Root Mean Squared Log Error (RMLSE)

Submissions are scored on the root mean squared log error (RMSLE) (the sklearn mean_squared_log_error with squared=False).

Submission File

For each id in the test set, you must predict the value for the target cost. The file should contain a header and have the following format:

    id,cost
    360336,99.615
    360337,87.203
    360338,101.111
    etc.


##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Media Campaign Cost Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Files

    train.csv - the training dataset; cost is the target
    test.csv - the test dataset; your objective is to predict cost
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, store_sales(in millions), unit_sales(in millions), total_children, num_children_at_home, avg_cars_at home(approx).1, gross_weight, recyclable_package, low_fat, units_per_case, store_sqft, coffee_bar, video_store, salad_bar, prepared_food, florist, cost
test.csv - column name: id, store_sales(in millions), unit_sales(in millions), total_children, num_children_at_home, avg_cars_at home(approx).1, gross_weight, recyclable_package, low_fat, units_per_case, store_sqft, coffee_bar, video_store, salad_bar, prepared_food, florist


## Dataset folder Location: 
../../kaggle-data/playground-series-s3e11. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
My solution is basically a blending of a total of 5 LightGBM (2x), XGBoost (2x) and Catboost (1x) models.

#### Data

    I added the original data to my cv-process. I used the original data for the training process of each fold, but validated exclusively on the synthetic competition data
    I therefore mark the original data with fold -1, so that it is not used as a validation dataset in any iteration
    I transformed the target variable with np.log(cost) and used rmse as objective
    I used a subset of the original variables: 'store_sqft', 'florist', 'salad_bar', 'prepared_food', 'coffee_bar', 'video_store', 'total_children', 'avg_cars_at home(approx).1', 'num_children_at_home'

#### Feature Engineering
    I spent a lot of time on feature engineering. The most important feature was an overall score for the store-specific attributes:
    store_features= ['coffee_bar', 'video_store', 'salad_bar', 'prepared_food', 'florist']
    df['store_score'] = df[stores_features].sum(axis=1)
    I also calculated the ratio in relation to the store size:
    df['store_score_ratio'] = df['store_sqft'] / df['store_score']
    With these two features, I was able to train a single-lightgbm model, which alone had a public score of 0.2926 and a private score of 0.29326 (private leaderboard range from 9 to 37)
    It was important to pass the new features to LGBM as categorical variables
    For XGBoost and Catboost, I was not able to get this significant improvement. I formed another feature for this and did not include the ratio:
    (df['florist']*3) + (df['food_proxy']*2) + df['coffee_bar'] + df['video_store']
    df['food_proxy'] is the sum of prepared food and salad bar and then captured at 1 and I have passed the feature as a numeric feature.
    The feature engineering was the biggest boost for me compared to the public solutions

#### Ensembling
Simple weighted blend optimized with Optuna


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: