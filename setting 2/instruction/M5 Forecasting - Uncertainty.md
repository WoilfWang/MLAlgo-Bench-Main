You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named M5_Forecasting_-_Uncertainty_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Note: This is one of the two complementary competitions that together comprise the M5 forecasting challenge. Can you estimate, as precisely as possible, the uncertainty distribution of the unit sales of various products sold in the USA by Walmart? This specific competition is the first of its kind, opening up new directions for both academic research and how uncertainty could be assessed and used in organizations. If you are interested in providing point (accuracy) forecasts for the same series, be sure to check out its companion competition.

How much camping gear will one store sell each month in a year? To the uninitiated, calculating sales at this level may seem as difficult as predicting the weather. Both types of forecasting rely on science and historical data. While a wrong weather forecast may result in you carrying around an umbrella on a sunny day, inaccurate business forecasts could result in actual or opportunity losses.  In this competition, in addition to traditional forecasting methods you’re also challenged to use machine learning to improve forecast accuracy.

The Makridakis Open Forecasting Center (MOFC) at the University of Nicosia conducts cutting-edge forecasting research and provides business forecast training. It helps companies achieve accurate predictions, estimate the levels of uncertainty, avoiding costly mistakes, and apply best forecasting practices. The MOFC is well known for its Makridakis Competitions, the first of which ran in the 1980s.

In this competition, the fifth iteration, you will use hierarchical sales data from Walmart, the world’s largest company by revenue, to forecast daily sales for the next 28 days and to make uncertainty estimates for these forecasts. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.

If successful, your work will continue to advance the theory and practice of forecasting. The methods used can be applied in various business areas, such as setting up appropriate inventory or service levels. Through its business support and training, the MOFC will help distribute the tools and knowledge so others can achieve more accurate and better calibrated forecasts, reduce waste and be able to appreciate uncertainty and its risk implications.


##  Evaluation Metric:
This competition uses a Weighted Scaled Pinball Loss (WSPL). Extensive details about the metric, scaling, and weighting can be found in the M5 Participants Guide.

Submission File

Similar to the point forecast competition, each row contains an id that is a concatenation of an item_id, a store_id, a quartile, and the prediction interval, which is either validation (corresponding to the Public leaderboard), or evaluation (corresponding to the Private leaderboard).

In addition, this competition has rows that have been aggregated at different levels. An X indicates the absence of a second aggregation level.

You are predicting 28 forecast days (F1-F28) of items sold for each row. For the validation rows, this corresponds to d_1914 - d_1941, and for the evaluation rows, this corresponds to d_1942 - d_1969. (Note: a month before the competition close, the ground truth for the validation rows will be provided.)

The files must have a header and should look like the following:

    id,F1,...F28
    Total_X_0.005_validation,53,...,201
    HOBBIES_1_001_CA_1_0.005_validation,0,...,2
    HOBBIES_1_002_CA_1_0.005_validation,2,...,11
    ...
    HOBBIES_1_001_CA_1_0.995_evaluation,3,...,7
    HOBBIES_1_002_CA_1_0.995_evaluation,1,...,4

##  Dataset Description:
In the challenge, you are predicting 9 quartiles of item sales at stores in various locations for two 28-day time periods. Information about the data is found in the M5 Participants Guide.

Files

    calendar.csv - Contains information about the dates on which the products are sold.
    sales_train_validation.csv - Contains the historical daily unit sales data per product and store [d_1 - d_1913]
    sample_submission.csv - The correct format for submissions. Reference the Evaluation tab for more info.
    sell_prices.csv - Contains information about the price of the products sold per store and date.
    sales_train_evaluation.csv - Includes sales [d_1 - d_1941] (labels used for the Public leaderboard)

sell_prices.csv - column name: store_id, item_id, wm_yr_wk, sell_price
calendar.csv - column name: date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type_1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI


## Dataset folder Location: 
../../kaggle-data/m5-forecasting-uncertainty. In this folder, there are the following files you can use: sell_prices.csv, sample_submission.csv, sales_train_validation.csv, sales_train_evaluation.csv, calendar.csv

## Solution Description:
### 1. Overview

* 2 neural network models with the same structure.
* One model predicts sales of individual items, and the other model predicts sales aggregated by all items, each category or each department.
* CV policies that go back 4 weeks or 8 weeks as one unit.
* Because of the seed problem, I did not use Validation Phase sales data to train the model which predicts sales of individual items.

### 2-2. Exploratory Data Analysis
There are some periods when items are not sold at all for many days. Even if a product is sold about 10 units on average, there are periods of time when it isn't sold at all for several months. For this reason, I thought it would be better to model the distribution of sales of individual items with a negative binomial distribution.

On the other hand, When sales are aggregated by all items, each category or each department, the distribution is quite different. The distribution seems a normal distribution. Therefore, I thought it would be better to model the distribution of sales aggregated by all items, each category or each department with a normal distribution or a Student's T-distribution.

In summary, I have decided to create the following two models.

* A negative binomial distribution model which predicts sales of individual items (I call it "Each Model" from here)
* A normal distribution or Student's T-distribution model which predicts sales aggregated by all items, each category or each department (I call it "Agg Model" from here)

### 2-3. Model Structure

This is a Long Short-Term Memory (LSTM)-based neural network architecture designed for predicting sales distribution over the next 28 days. Here’s a detailed description:
	1.	Input Features:
	•	Sales, Calendar, Prices in Recent 28 Days: Historical data of sales, calendar information (e.g., day of the week), and prices for the past 28 days are used as input.
	•	Calendar, Prices for Next 28 Days: Calendar and price information for the upcoming 28 days are also provided as input for prediction.
	2.	Embedding Layer:
	•	An embedding layer encodes categorical features (e.g., day of the week, holiday information) into dense, continuous representations to be processed further by the LSTM layers.
	3.	LSTM Layers:
	•	Two LSTM layers are stacked:
	•	LSTM Layer #1: Processes the time-series input from the recent 28 days.
	•	LSTM Layer #2: Builds on the output of the first layer to capture more complex temporal dependencies.
	4.	Fully Connected (FC) Layers:
	•	The outputs of the LSTM layers are passed to a series of Fully Connected (FC) layers, each dedicated to predicting the sales distribution for a specific day of the week (e.g., FC Mon for Monday, FC Tue for Tuesday, etc.).
	5.	Output:
	•	The network predicts the parameters of the sales distribution for each of the 28 days being forecasted. These parameters could represent, for example, the mean and variance of predicted sales.

This model is a neural network which takes features of sales, calendar and prices in recent 28 days and features of calendar and prices in 28 days to predict as input, and outputs parameters of sales distribution in 28 days to predict.
It consists of an embedding layer that encodes categorical features, two LSTM layers that process time series data, and 7 fully-connected layers for each day of the week.
Both "Each Model" and "Agg Model" have the same structure except that "Agg Model" does not use features of prices.

The final output layer was originally LSTM, but that didn't seem to be able to express the periodicity of day of the week well, so I made a dedicated fully-connected layer for each day of the week.

The parameters of the neural network are obtained by maximum likelihood estimation.
In other words, I compute the negative log-likelihood of actual sales in the distribution predicted by the model and optimize it so that it is minimized.

### 4. Feature Engineering

I did not do much feature engineering because I used the model structure shown above. The features I created are whether it is Christmas, whether it is a weekday holiday, the price trend (correlation coefficient with the date), etc. I created a model with LightGBM to predict sales using the created features as input, and I selected features by looking at the model's importance and Permutation Importance.

I used PowerTransformer to bring the sales data of each time series close to a normal distribution for "Each Model".
On the other hand, for "Agg Model" I used raw sales data. I didn't preprocess them.

For example, a product that is sold one unit on average should have a very different distribution comparing with another one  that is sold ten units on average. I believe that adjusting the scale would greatly disrupt the shape of such a distribution, so I decided to do nothing. The other reason is that I wanted to put out the nine-quantile point as a discrete integer variable.

### 5. Cross Validation

In this competition I was worried about how to get the score of local Cross Validation.
Using TimeSeriesSplit of scikit-learn, for example, if n_splits=4, the training data for 5 years is divided into 5 equal parts, and the training data for the first fold has only 1 year data.
In that case, many products have not been sold yet, and it may be very different from the latest distribution, so it is not very useful.

Therefore, I decided to go back to the past with the time series data for each fold in 1 unit of 4 weeks. And I used the last 4 weeks as test data, the previous 4 weeks as validation data, and all remaining data as training data.

For example, in the case of the first fold when performing 4 CVs, the last 12 weeks of the training data of 5 years are removed. I use the last 4 weeks of the remaining data as test data, the previous 4 weeks as validation data, and the rest as training data.
The next fold removes the last 8 weeks and splits into testing, validation and training.
The third fold removes the last 4 weeks. The last fold uses all data.

At first, I was proceeding with this policy, but the period of validation data and test data was too short for 4 weeks.
The epoch of early stopping at each fold was completely different,
and the score at each fold was also completely different.
Probably because the distribution differs greatly every four weeks.
At the last minute, I changed the CV split policy from 1 unit of 4 weeks to 1 unit of 8 weeks.
However, the situation didn't change much.

I don't know if this method was correct.

I evaluated scores of diffrent models by using weighted loss of test data in Local CV.
Weights were calculated by sales in last 28 days of training data.
This made it possible to confirm whether the forecast was correct for time series data of higher importance. I didn't implement WSPL, which is the evaluation index of the competition, because it took a long time to process and it was troublesome.

### 6. Seed Problem

Another thing I was worried about was that I had a problem that the score changed greatly depending on the seed.
"Agg Model" is prone to over fitting because there are few variations in time series data, and that probably caused the seed problem.
I tried to reduce the number of nodes in the hidden layer, put dropout, put weight_decay, and various over-fitting measures, but until the end, this problem was not solved.

The annoyance of this problem is that it's hard to distinguish whether the improvement in score is due to my efforts or just a good seed.
Therefore, when performing a hyperparameter search, I compared the average values of the results from multiple different seeds.

When selecting the final submission,
I decided not to use the published validation phase sales data (from April 25 to May 22, 2016) as one of training datasets, and I created models from various seeds.
Then, I chose the one with the highest Public Leaderboard score I tried.
The decision not to use the validation phase sales data for training was a big challenge, but as a result, this policy allowed me to select a good seed.

This problem was not occurred in "Each Model".
So, I trained "Each Model" using validation phase sales data.

### 7. Target Encoding

As mentioned above, "Each Model" did not transform the sales scale as a preprocess.
I was worried that I could make accurate predictions for products with particularly large sales.
So I added target encodings so that "Each Model" can use it to adjust the scale.

I decided to find the average value of sales for each time series and day of the week, and enter it before the final fully-connected layer.
To avoid the leak, I should use only the past data to calculate target encodings, but since there was no time, I did not take such a consideration and made it the average of all periods.

As a result, this feature was also useful for "Agg Model".

### 8. Other Details

To prioritize the most recent data, I oversampled the 2015 data by a factor of 2 and the 2016 data by a factor of 4.

Since the evaluation index is weighted by sales amount, I tried to weight loss by each time series data, which was effective in the "Each Model", but was not effective in the "Agg Model".


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: