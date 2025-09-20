You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Regression_of_Used_Car_Prices_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description

Welcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: The goal of this competition is to predict the price of used cars based on various attributes.

##  Evaluation Metric:
Root Mean Squared Error (RMSE)
Submissions are scored on the root mean squared error.

Submission File

For each id in the test set, you must predict the price of the car. The file should contain a header and have the following format:

    id,price
    188533,43878.016
    188534,43878.016
    188535,43878.016
    etc.

##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Used Car Price Prediction Dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Files

    train.csv - the training dataset; price is the continuous target
    test.csv - the test dataset; your objective is to predict the value of price for each row
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, brand, model, model_year, milage, fuel_type, engine, transmission, ext_col, int_col, accident, clean_title, price
test.csv - column name: id, brand, model, model_year, milage, fuel_type, engine, transmission, ext_col, int_col, accident, clean_title


## Dataset folder Location: 
../../kaggle-data/playground-series-s4e9. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Did I really made it to the top? I am still surprised and excited.

Way to the solution: I spent the first two weeks with reading the discussions, playing around with catboost and publishing an ensemble. The plan was to collect diverse models and ensemble them with Ridge, with the same pipeline I used in this notebook. The final ensemble, which I chose as my first final submission, would have landed me on the second place and differed from the notebook in the following points:

    I used 20 cv folds
    I included original data in some of the models (even two times in LGBM)
    I did compute a SVR with a rbf kernel as suggested by broccoli beef in this discussion post instead of the linear SVR.
    I included all categorical features additionally as target encoded to catboost, but I used the median, not the mean for target encoding. I did this leakfree, meaning that I recomputed the targetencoded columns in each fold. Moreover, I used Catboost as classifier, not as regressor. Catboost predicted the outlier prices (see function bin_price). The hyperparameters were found by optuna. The oof predictions were not used in the ensemble. They were used as an additional feature in a LGBM (or the NN for my second final submission).

def bin_price(data):
    df = data.copy()
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(df['price'], 25)
    Q3 = np.percentile(df['price'], 75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df['price'] > upper_bound)]
    df['price_bin'] = (df['price'] < upper_bound).astype(int)

    return df

cat_params2 = {
    'early_stopping_rounds':25,
    'use_best_model': True,
    "verbose": False ,
    'cat_features': cat_cols,
    'min_data_in_leaf': 16, 
    'learning_rate': 0.03355311405703999, 
    'random_strength': 11.663619399375248, 
    'l2_leaf_reg': 17.703146378123996, 
    'max_depth': 10, 
    'subsample': 0.9479174100256215, 
     'border_count': 130, 
    'bagging_temperature': 24.032067560148384
}

I included the catboost oof predictions as an additional feature for LGBM

I used a second LGBM (LGBM5), where I label encoded all categorical data (rare categories summarized in a category "rare" as done with the NN in the notebook) and raised max_bin.

lgb_params = {
    'verbose' : -1,
    'early_stopping_rounds':25,
    'loss_function':"RMSE",
    'n_estimators': 2000, 
    'max_bin': 30000,
}

I included fastai computations from Autogluon (with a nested cv over 20 folds to be 100% leakfree)

 predictor = TabularPredictor(label='price',
                             eval_metric='rmse',
                             problem_type="regression").fit(X_train,
                                                       pseudo_data = data_original, 
                                                       num_bag_folds = 10,
                                                       num_bag_sets = 2,
                                                       time_limit=1800,
                                                       included_model_types = ['FASTAI'], 
                                                       keep_only_best = True,
                                                       presets="best_quality",
                                                      )

I ended up with a crossvalidation score of 72300 and the following models (_st means that the catboost oofs are included):

    •   LGBM5_st: The largest segment, accounting for 32.1% of the weights.
	•	LGBM1_st: The second-largest segment, with 19.8% of the weights.
	•	SVR: Representing 18.4% of the weights.
	•	LGBM1: Covering 11.9% of the weights.
	•	Fastai: Allocated 10.0% of the weights.
	•	NN1: The smallest segment, comprising 7.7% of the weights.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: