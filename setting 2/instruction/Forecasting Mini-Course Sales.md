You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Forecasting_Mini-Course_Sales_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the 2023 edition of Kaggle's Playground Series!

Thank you to everyone who participated in and contributed to Season 3 Playground Series so far! 

With the same goal to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science, we will continue launching the Tabular Tuesday in July every Tuesday 00:00 UTC, with each competition running for 3 weeks. Again, these will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc. 

Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

##  Evaluation Metric:
Submissions are evaluated on SMAPE between forecasts and actual values. We define SMAPE = 0 when the actual and predicted values are both 0.

Submission File

For each id in the test set, you must predict the corresponding num_sold. The file should contain a header and have the following format:

    id,num_sold
    136950,100
    136950,100
    136950,100
    etc.


##  Dataset Description:
For this challenge, you will be predicting a full year worth of sales for various fictitious learning modules from different fictitious Kaggle-branded stores in different (real!) countries. This dataset is completely synthetic, but contains many effects you see in real-world data, e.g., weekend and holiday effect, seasonality, etc. You are given the task of predicting sales during for year 2022.
Good luck!

Files

    train.csv - the training set, which includes the sales data for each date-country-store-item combination. 
    test.csv - the test set; your task is to predict the corresponding item sales for each date-country-store-item combination. Note the Public leaderboard is scored on the first quarter of the test year, and the Private on the remaining.
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, date, country, store, product, num_sold
test.csv - column name: id, date, country, store, product


## Dataset folder Location: 
../../kaggle-data/playground-series-s3e19. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
#### Correcting the trend

Based on the plot of daily total sales, it's obvious that trend in 2020 is abnormal.
My method to correct the trend:

    Calculate daily fraction of each of the 75 categories.
    Use STL decomposition to extract the trend from daily total sales series.
    Replace the trend from 2020-02-15 to 2020-11-15 with the average trend of the other years.
    Restore the daily total sales series.
    Restore the series of each category based on its fraction.

#### Modelling the baselines
Baseline for total daily sales seems to be discrete on each year. The baseline of each category can be modeled as:
$$B_{y, p, c, s} = B_y F_{y, p} F_{y, c} F_{y, s}$$
where B is baseline level, F is fraction, y is year, p is product, c is country, s is store. The subscript denotes the corresponding combined categories. So to find baselines of 2022 one needs to find all these values with y=2022.

Fraction of country by year

Country fractions fluctuate with time. It turned out that these fractions are correlated with GDP per capita. However for unknown reason all countries in 2022 have the same fraction, i.e., 0.2. Probably due to data construction error.
$$F_{y, c} = f_{y, c}; F_{2022, c} = 0.2$$
where f denotes empirical fraction and y = 2017, 2018 … 2021.

Fraction of product by year

Product fractions are roughly constant through years. There seems to be some seasonality of two-year period. But I'd rather capture that by temporal embedding. Here I set them constant as weighted average:
$$F_{2022, p} = F_{y, p} = (2(f_{2017, p} + f_{2019, p} + f_{2021, p}) + 3(f_{2018, p} + f_{2020, p}))/12$$

Fraction of store by year

Store fractions seems to be constant through years. The fluctuation is negligible.
$$F_{2022, s} = F_{y, s} = f_{s}$$

Baseline of total daily sales.

I used median as baseline. For 2022 I left it unknown and determined it by LB probing at the end.
$$B_y = median_y; B_{2022} \approx 18900$$

#### EDA by category
Standardization

With the baseline model, standardization can be done directly:
$$S_{y, p, c, s, d} =  N_{y, p, c, s, d} /  F_{y, p}  /  F_{y, c} / F_{y, s} / B_y - 1$$
where N is num of sold, S is standardized num of sold, d is date.

#### Sum the store sales
Reason to sum the store sales:

    Based on the analysis above, store fractions seem to be constant. And there is no difference of seasonality and holiday effect among stores.
    For some categories, e.g. Using LLMs to Win Friends and Influence People - Argentina - Kaggle Learn, the num of sold is too small and it oscillates between several integers. Such series may affect model training.

So I used the sum of daily sales by product and country for training and validation. Similar standardization applied:
$$S_{y, p, c, d} =  N_{y, p, c, d} /  F_{y, p}  /  F_{y, c} / B_y - 1$$
For prediction, convert this intermediate value back to num of sold:
$$N_{y, p, c, s, d} =  (S_{y, p, c, d} + 1) B_y F_{y, p} F_{y, c} F_{y, s}$$
I put all model parameters to an excel table. See attachment.

#### Training and prediction
Pipeline

Since seasonalities are different among products, I trained each product separately. So there are 5 models.
Training data is split by year to 5 folds, followed by cross validation.

Feature engineering

Since private score is based on the last 75% of 2022, lag features are not needed.

Temporal embedding

    For week seasonality, I used Fourier terms up to order 3 .
    For two-year seasonality, I used Fourier terms up to order 5 .

Holiday and date related features

I generated two categorical features:

    c_holiday: country + corresponding holidays
    c_year_date: country + 365 dates of a year

These categorical features are then converted to numeric levels by target encoding. I used linear mixed model with S ~ temporal variables grouped by the categorical variable. Of course, encoding is done separately for each product. To avoid overfitting, encoding is done with 4~5 fold OOS predictions.

Modeling

I used lightbm with linear tree and dart. I didn't do much hyper-parameter tuning except I set num_leaves to a small value (~5). The optimal num_iterations is determined by the cross-validation RMSE. The final model is trained on the whole training set with optimal num_iterations.
I didn't do any ensmeble.



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: