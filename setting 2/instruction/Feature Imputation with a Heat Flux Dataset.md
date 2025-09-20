You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Feature_Imputation_with_a_Heat_Flux_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the 2023 edition of Kaggle's Playground Series!

Thank you to everyone who participated in and contributed to Season 3 Playground Series so far! 

With the same goal to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science, we will continue launching the Tabular Tuesday in May every Tuesday 00:00 UTC, with each competition running for 2 weeks. Again, these will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.. 

Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

##  Evaluation Metric:
Root Mean Squared Error (RMSE)

Submissions are scored on the root mean squared error. RMSE is defined as:
$$
\textrm{RMSE} =  \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 }
$$
where \( \hat{y}_i \) is the predicted value and \( y_i \) is the original value for each instance \(i\).

Submission File

This is an imputation problem. You are to predict the missing values of the feature x_e_out [-] (with the corresponding row id). The file should contain a header and have the following format:

    id,x_e_out [-]
    4,0.00
    7.0.12
    10,-0.02
    etc.


##  Dataset Description:

The dataset for this competition (both train and test) was generated from a deep learning model trained on the Predicting Critical Heat Flux dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Files

    data.csv - the competition dataset; your objective is to impute the missing values of the feature x_e_out [-] (equilibrium quality) 
    sample_submission.csv - a sample submission file in the correct format

data.csv - column name: id, author, geometry, pressure, mass_flux, x_e_out [-], D_e, D_h, length, chf_exp


## Dataset folder Location: 
../../kaggle-data/playground-series-s3e15. In this folder, there are the following files you can use: sample_submission.csv, data.csv

## Solution Description:
#### Pre-Processing
The initial steps involved a lot of manual work going through the data and looking for patterns to solve the easier missing values. This filled out most of the values in columns D_e, D_h, and length. A small sample of that code for author ‘Inasaka’ rows is below

data.loc[(data['author'] == 'Inasaka') & ((data['D_h'] == 3.0) | (data['length'] == 100.0)) & (data['D_e'].isnull()), 'D_e'] = 3

data.loc[(data['author'] == 'Inasaka') & ((data['D_e'] == 3.0) | (data['length'] == 100.0)) & (data['D_h'].isnull()), 'D_h'] = 3

data.loc[(data['author'] == 'Inasaka') & ((data['D_e'] == 3.0) | (data['D_h'] == 3.0)) & (data['length'].isnull()), 'length'] = 100

My next step of pre-processing was using the K-nearest neighbors imputer to fill in the remaining holes in pressure, mass flux, and whatever was remaining in D_e, D_h, and length. Thanks to @validmodel for this post which laid out many different imputation techniques. This ultimately led me to choosing sklearn's KNN imputer to finish the remaining numerical values. I tried several different values for the n_neighbors parameter and found I was getting the best CV scores for my models in the 87 to 89 range.

The missing categorical columns were imputed with some more manual code, such as below

data.loc[(data['author'] == 'Inasaka') & ((data['D_h'] == 3.0) | (data['length'] == 100.0)) & (data['D_e'].isnull()), 'D_e'] = 3

data.loc[(data['author'] == 'Inasaka') & ((data['D_e'] == 3.0) | (data['length'] == 100.0)) & (data['D_h'].isnull()), 'D_h'] = 3

data.loc[(data['author'] == 'Inasaka') & ((data['D_e'] == 3.0) | (data['D_h'] == 3.0)) & (data['length'].isnull()), 'length'] = 100

This did reasonably well and I was getting solid CV scores but not the strongest public LB scores (more on this later).

However, near the end of the competition @shalfey made this excellent post. I placed all of this code to the front of my pre-processing. It led to a noticeable improvement in my scores and I most certainly would not have finished quite as high as I did without it.

#### Model Building and Cross Validation
I first built a basic untuned XGBoost model to get a baseline CV score. I was very surprised when it came in at around 0.07305. That score was much better than the 1st place score of the public LB (around 0.0743 at the time). This caused me a sense of dread 😬.

It was extremely unlikely that this basic model would not only put me in 1st but put me in 1st by a huge margin. I continued forward anyways and tuned the model up, made some test predictions, and submitted. The public score came to 0.07555.

There were two possible reasons for the gap between my CV score and the public LB.

    My CV was flawed somewhere (I strongly suspected that my imputation pre-processing strategy had caused target leakage)
    The 20% of data used for the public LB contained an abnormal section of data that my model didn’t do great on but would perform closer to my CV over the entire test set.

I was pretty sure the problem was number 1. I spent a lot of time changing up my workflow to eliminate any possibility of target leakage. However, my CV scores were still coming out to around 0.0732. I was also scoring much worse on my public LB scores when submitting those models. These results were starting to convince me that reason number 2 was actually true.

Based on this, I went back to my original workflow of doing all pre-processing and imputation ahead of my cross validation instead of doing it fold by fold. In this case, there was a tradeoff between slight target leakage (since the KNN Imputer utilized the target feature) and increasing the data quality/accuracy of input features for model training.

If I was setting up a real world cross validation experiment the correct choice would be to do all imputing on a fold by fold basis and gather more data if necessary to improve results. However, in this case, gathering more data is obviously not an option. I was limited to the competition/original datasets and needed to squeeze every bit of information out of them to maximize the accuracy of the KNN Imputer step. Even if doing so caused a tiny bit of target leakage.

#### Utilizing the Original Dataset

One unique feature of doing these playground series competitions is that all of them use synthetic data generated from an original dataset. This always leads to the question “What do we do with the original data?” Based on the playground competitions I have participated in, the answer is that you always need to find a way to utilize it.

I have to credit @adaubas for this post during the blueberry yield competition for really getting me to think about how the original dataset should be used. I used to just treat it as simply additional data and immediately join it to the train dataset. But, I have found that is not the best way to handle it.

The problem with doing this is that it will change the distribution of your OOF validation data when doing CV. The synthetic data seems to always be much noisier than the original data. This post from @sergiosaharovskiy is a great visualization of the observed phenomenon. Including the original data in the OOF validation data will give you a better CV score than what you can expect when using your model to predict purely synthetic test data.

I have found the best way to use the original data is to keep it separated from your train data and then concat the entire original data back to each fold in your cross validation. Example code below where X_original and y_original are the entire original dataset.

```python
kf = KFold(n_splits=10, random_state=8, shuffle=True)

for train_idx, val_idx in kf.split(X_tr, y_tr):
    X_t, X_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
    y_t, y_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

    X_train = pd.concat([X_t, X_original], ignore_index = True)
    y_train = pd.concat([y_t, y_original], ignore_index = True)

    model = LGBMRegressor()

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    y_pred = model.predict(X_val)

    score = mean_squared_error(y_val, y_pred, squared=False)
```
#### Models and Ensemble
I ended up tuning 4 models that I was happy with and wanted to see how they would perform as an ensemble.

Model 1: LightGBM using all features of the original dataset except for geometry

Model 2: LightGBM same as model 1 but also added two new features ‘adiabatic_surface_area’ and ‘surface_diameter_ratio’. Credit to @tetsutani and his notebook for the feature ideas

Model 3: LightGBM same as model 2 but also added a single feature/component using PLSRegression on features mass_flux, pressure, and chf_exp. Once again thanks to @adaubas for this post which led me to learning about PLS.

Model 4: XGBoost same features as model 1

I had originally planned to explore @tetsutani’s notebook more and try out LAD regression and hill climbing from @samuelcortinhas notebook here. Unfortunately, I simply ran out of time and my ensemble ended up just being 4 basic weights for each model.
#### Conclusion
One thing that this competition further reinforced in me is that the most important thing to strive for is getting a CV workflow that you trust in. Let your CV scores guide your decision making and don’t place too much weight on the public LB scores.

Once again, thanks to everyone in the community for being so welcoming and eager to share their knowledge! I have learned so much! Best of luck in your future competitions!


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: