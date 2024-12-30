You are given a detailed description of a data science competition task, including the evaluation metric and a detailed description of the dataset. You are required to complete this competition using Python. 
Additionally, a general solution is provided for your reference, and you should implement this solution according to the given approach. 

You may use any libraries that might be helpful.
Finally, you need to generate a submission.csv file as specified.

## Task Description
Welcome to the 2023 edition of Kaggle's Playground Series!

Thank you to everyone who participated in and contributed to Season 3 Playground Series so far! 

With the same goal to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science, we will continue launching the Tabular Tuesday in June every Tuesday 00:00 UTC, with each competition running for 2 weeks. Again, these will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc. 

**Synthetically-Generated Datasets**
Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

##  Evaluation Metric:

Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Submission File
For each id in the test set, you must predict the probability of a Machine failure. The file should contain a header and have the following format:

    id,Machine failure
    136429,0.5
    136430,0.1
    136431,0.9
    etc.
    

##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Machine Failure Predictions. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance. In this competition, our task is to predict machine failure based on 13 features:

    Product ID
    Type
    Air temperature
    Process temperature
    Torque
    Rotational speed
    Tool wear
    TWF
    HDF
    PWF
    OSF
    RNF

**Files**

    train.csv - the training dataset; Machine failure is the (binary) target (which, in order to align with the ordering of the original dataset, is not in the last column position)
    test.csv - the test dataset; your objective is to predict the probability of Machine failure
    sample_submission.csv - a sample submission file in the correct format

## Dataset folder Location: 
../../kaggle-data/playground-series-s3e17.

## Solution Description:
The following is the solution of datadote team, who ranks at the 11th position with the score of 0.98429.

I already expected that the top places will get few shake-up but getting myself up to top 1% despite telling all my approaches? That's different story. Anyway, here's what I've done to get to 11th position

### Cross-Validation Process
I used MultilabelStratifiedKFold from iterative-stratification library because I want to make sure that the percentage for all types of failure is kept throughout the fold. I also explained this here. The amount of split I used was 10 folds, because 5 folds isn't correlated enough with the public LB.

I always try to do everything inside cross-validation pipeline, such as adding the original data, encoding, scaling, etc. This way, I don't have to worry a lot about leakage.

### Feature Engineering
I use category-encoders' CatBoost Encoder in all my models, except CatBoost itself, since I noticed that CatBoost performs exceptionally well for some reason. I've already posted about this here actually, the main difference is that I put it inside the model pipeline instead of using it before doing a CV, thus preventing major leakage. Example code is as follows:

    Encoder = CatBoostEncoder(cols = ['Product ID', 'Type'])
    model = make_pipeline(Encoder, model)

As for creating new features, I've only created one: IsTrouble. This feature describes whether there is any type of failure (TWF, HDF, etc.) that is happening. I only use this feature in one model: Gaussian Naive Bayes. The rest is just encoding.

### Tuning and Ensembling
I used 6 models: Gaussian Naive Bayes, Random Forest, XGBoost, LightGBM, LightGBM's Dart, and CatBoost. For gradient boosting models, I used Optuna to tune them. For Naive Bayes and Random Forest, I didn't tune them at all, only calibrated them (I also posted about it here), and did further preprocessing for Naive Bayes (such as scale normalization and Yeo-Johnson transformation).

Once I've done building all of them, I used LogisticRegression to find optimal weight for my voting ensemble. The CV score is 0.98006. Finally, I retrained my ensemble on the whole dataset for better final result instead of just relying on the CV process.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: