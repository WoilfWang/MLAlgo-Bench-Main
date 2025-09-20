You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Walmart_Recruiting_-_Store_Sales_Forecasting_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
One challenge of modeling retail data is the need to make decisions based on limited history. If Christmas comes but once a year, so does the chance to see how strategic decisions impacted the bottom line.

In this recruiting competition, job-seekers are provided with historical sales data for 45 Walmart stores located in different regions. Each store contains many departments, and participants must project the sales for each department in each store. To add to the challenge, selected holiday markdown events are included in the dataset. These markdowns are known to affect sales, but it is challenging to predict which departments are affected and the extent of the impact.

Want to work in a great environment with some of the world's largest data sets? This is a chance to display your modeling mettle to the Walmart hiring teams.

This competition counts towards rankings & achievements.  If you wish to be considered for an interview at Walmart, check the box "Allow host to contact me" when you make your first entry. 

You must compete as an individual in recruiting competitions. You may only use the provided data to make your predictions.

##  Evaluation Metric:
This competition is evaluated on the weighted mean absolute error (WMAE):

$$\textrm{WMAE} = \frac{1}{\sum{w_i}} \sum_{i=1}^n w_i | y_i - \hat{y}_i |$$
where

    n is the number of rows
    \\( \hat{y}_i \\) is the predicted sales
    \\( y_i \\) is the actual sales
    \\( w_i \\) are weights. w = 5 if the week is a holiday week, 1 otherwise

Submission File

For each row in the test set (store + department + date triplet), you should predict the weekly sales of that department. The Id column is formed by concatenating the Store, Dept, and Date with underscores (e.g. Store_Dept_2012-11-02).  The file should have a header and looks like the following:

    Id,Weekly_Sales
    1_1_2012-11-02,0
    1_1_2012-11-09,0
    1_1_2012-11-16,0
    ...

##  Dataset Description:
You are provided with historical sales data for 45 Walmart stores located in different regions. Each store contains a number of departments, and you are tasked with predicting the department-wide sales for each store.

In addition, Walmart runs several promotional markdown events throughout the year. These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks. Part of the challenge presented by this competition is modeling the effects of markdowns on these holiday weeks in the absence of complete/ideal historical data.

stores.csv

This file contains anonymized information about the 45 stores, indicating the type and size of store.

train.csv

This is the historical training data, which covers to 2010-02-05 to 2012-11-01. Within this file you will find the following fields:

    Store - the store number
    Dept - the department number
    Date - the week
    Weekly_Sales -  sales for the given department in the given store
    IsHoliday - whether the week is a special holiday week

test.csv

This file is identical to train.csv, except we have withheld the weekly sales. You must predict the sales for each triplet of store, department, and date in this file.

features.csv

This file contains additional data related to the store, department, and regional activity for the given dates. It contains the following fields:

    Store - the store number
    Date - the week
    Temperature - average temperature in the region
    Fuel_Price - cost of fuel in the region
    MarkDown1-5 - anonymized data related to promotional markdowns that Walmart is running. MarkDown data is only available after Nov 2011, and is not available for all stores all the time. Any missing value is marked with an NA.
    CPI - the consumer price index
    Unemployment - the unemployment rate
    IsHoliday - whether the week is a special holiday week

For convenience, the four holidays fall within the following weeks in the dataset (not all holidays are in the data):
Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13Labor Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13

train.csv - column name: Store, Dept, Date, Weekly_Sales, IsHoliday
stores.csv - column name: Store, Type, Size
test.csv - column name: Store, Dept, Date, IsHoliday
features.csv - column name: Store, Date, Temperature, Fuel_Price, MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5, CPI, Unemployment, IsHoliday


## Dataset folder Location: 
../../kaggle-data/walmart-recruiting-store-sales-forecasting. In this folder, there are the following files you can use: train.csv, stores.csv, sampleSubmission.csv, test.csv, features.csv

## Solution Description:
Congrats David and all the contestants.

Personally, I would like to personally take this opportunity to  thank  Kaggle community, It has been an enjoyable experience. I have learnt a lot from everyone in this website.

With regards to my model, I'll upload a detailed explanation on "how and why" of my approach. I used a hybrid approach of statistical and machine learning methods.

I used SAS (for data prep/ARIMA/UCM) and R (for the remainder models) together. I used weighted average and trimmed mean of following  6 methods. The goal from  the beginning was to build a robust model that will be able to withstand uncertainty.

Statistical Methods:

1. Auto-regressive Integrated Moving Average (ARIMA)

2. Unobserved Components Model (UCM)

Machine Learning Methods:

3. Random Forest

4. Linear Regression

5. K nearest regression

6. Principle Component Regression

My model did not use any features. I simply used past values to predict future values. 

With Regards to variables (features) I used week of the year (1 thru 52), this would capture almost all the lag and lead effects of holidays except for new year which was moving and one other holiday. I built individual models for each department. I weighted holidays for stores with high growth rate vs. prev year differently than the stores without high growth.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: