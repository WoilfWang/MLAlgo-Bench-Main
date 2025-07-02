You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named COVID19_Global_Forecasting_(Week_5)_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
This is week 5 of Kaggle's COVID-19 forecasting series, following the Week 4 competition. This competition has some changes from prior weeks - be sure to check the Evaluation and Data pages for more details. All of the prior discussion forums have been migrated to this competition for continuity.

#### Background
The White House Office of Science and Technology Policy (OSTP) pulled together a coalition research groups and companies (including Kaggle)  to prepare the COVID-19 Open Research Dataset (CORD-19) to attempt to address key open scientific questions on COVID-19. Those questions are drawn from National Academies of Sciences, Engineering, and Medicine’s (NASEM) and the World Health Organization (WHO).

#### The Challenge
Kaggle is launching a companion COVID-19 forecasting challenges to help answer a subset of the NASEM/WHO questions. While the challenge involves developing quantile estimates intervals for confirmed cases and fatalities between May 12 and June 7 by region, the primary goal isn't only to produce accurate forecasts. It’s also to identify factors that appear to impact the transmission rate of COVID-19.

You are encouraged to pull in, curate and share data sources that might be helpful. If you find variables that look like they impact the transmission rate, please share your finding in a notebook. 

As the data becomes available, we will update the leaderboard with live results based on data made available from the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE).

We have received support and guidance from health and policy organizations in launching these challenges. We're hopeful the Kaggle community can make valuable contributions to developing a better understanding of factors that impact the transmission of COVID-19. 

This is a Code Competition. Refer to Code Requirements for details.

##  Evaluation Metric:
Public and Private Leaderboard

To have a public leaderboard for this forecasting task, we will be using data from 7 days before to 7 days after competition launch. Only use data prior to 2020-04-27 for predictions on the public leaderboard period. Use up to and including the most recent data for predictions on the private leaderboard period.

Public Leaderboard Period: 2020-04-27 - 2020-05-11
Private Leaderboard Period: 2020-05-13 - 2020-06-10

Evaluation
Submissions are scored using the Weighted Pinball Loss.
$$
\text{score} =  \frac{1}{N_{f}} \sum_{f} w_{f} \frac{1}{N_{\tau}} \sum_{\tau} L_{\tau}(y_i,\hat{y}_{i})
$$
where:
$$
\begin{eqnarray}
 L_{\tau}(y,\hat{y}) & = & (y - \hat{y}) \tau & \textrm{ if } y \geq \hat{y} \\
& = & (\hat{y} - y) (1 - \tau) & \textrm{ if } \hat{y} > y
\end{eqnarray}
$$
and:

\(y \) is the ground truth value
\(\hat{y} \) is the predicted value
\(\tau \) is the quantile to be predicted, e.g., one of [0.05, 0.50, 0.95]
\(N_{f}\) is the total number of forecast (\(f\)) day x target combinations
\(N_{\tau} \) is the total number of quantiles to predict
\( w\) is a weighting factor

Weights are calculated as follows:

ConfirmedCases:  \(\log(\text{population}+1)^{-1}\)
Fatalities:  \(10  \cdot \log(\text{population}+1)^{-1}\)

Submission File

For each ForecastId in the test set, you'll predict the 0.05, 0.50, and 0.95 quantiles for daily COVID-19 cases and fatalities to date. The file should contain a header and have the following format:

    ForecastId_Quantile,TargetValue
    1_0.05,1
    1_0.50,1
    1_0.95,1
    2_0.05,1
    etc.

You will get the ForecastId_Quantile for the corresponding date and location from the test.csv file.

##  Dataset Description:

In this challenge, you will be predicting the daily number of confirmed COVID19 cases in various locations across the world, as well as the number of resulting fatalities, for future dates. This latest challenge includes US state county data.

Files

    train.csv - the training data (you are encouraged to join in many more useful external datasets)
    test.csv - the dates to predict; there is a week of overlap with the training data for the initial Public leaderboard. Once submissions are paused, the Public leaderboard will update based on last 28 days of predicted data.
    submission.csv - a sample submission in the correct format; again, predictions should be daily

Data Source

    This evaluation data for this competition comes from John Hopkins CSSE, which is uninvolved in the competition. 
    See their README for a description of how the data was collected.
    They are currently updating the data daily.

train.csv - column name: Id, County, Province_State, Country_Region, Population, Weight, Date, Target, TargetValue
test.csv - column name: ForecastId, County, Province_State, Country_Region, Population, Weight, Date, Target


## Dataset folder Location: 
../../kaggle-data/covid19-global-forecasting-week-5. In this folder, there are the following files you can use: train.csv, submission.csv, test.csv

## Solution Description:
Summary
LGBMs with quantile regression were trained on time-series and geo features for short term predictions. Aggressive 1w average smoothing was used for long term predictions. Due to the large number of small locations the top 30 country/state had to be adjusted manually.

Feature Extraction

    Population
    Latitude, Longitude
    Day of week
    Share of total cases for each day of week
    Rolling mean/std for 1w, 2w, 3w
    Cumulative totals
    Confirmed - Fatality rate
    Trend from last 2-3 weeks
    Normalized features by population
    Nearby features based on the closest 5-10-20 locations


Rescaled and rounded features to 1-2 decimals to decrease overfitting

Modeling

For each target/quantile/forecast lag separate model was trained  with location based 5-fold CV and early stopping based on pinball loss.

Models were only trained to predict the next 1-14 days.

    Trained bunch of LGBMs with random parameters to blend
    Sample Weighting based on location weights and time decay 

Post processing

    Clipped negative predictions at 0 
    Made sure the 0.05 (0.95) quantile predictions are not higher (lower) than the median
    Smooth daily predictions (Y[t] * 0.66 + Y[t-1] * 0.33)
    For US country total used the state level rollups for median
    Manually inspected and adjusted the top 30 countries
    Flat long-term predictions based on the last predicted weekly average
    Small daily decay was added to 0.05 quantile and median.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: