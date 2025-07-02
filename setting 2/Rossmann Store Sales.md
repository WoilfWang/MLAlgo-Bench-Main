You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Rossmann_Store_Sales_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

In their first Kaggle competition, Rossmann is challenging you to predict 6 weeks of daily sales for 1,115 stores located across Germany. Reliable sales forecasts enable store managers to create effective staff schedules that increase productivity and motivation. By helping Rossmann create a robust prediction model, you will help store managers stay focused on what’s most important to them: their customers and their teams!

##  Evaluation Metric:
Submissions are evaluated on the Root Mean Square Percentage Error (RMSPE). The RMSPE is calculated as
$$\textrm{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2},$$

where y_i denotes the sales of a single store on a single day and yhat_i denotes the corresponding prediction. Any day and store with 0 sales is ignored in scoring.

#### Submission File
The file should contain a header and have the following format:

    Id,Sales
    1,0
    2,0
    3,0
    etc.

##  Dataset Description:
You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.
#### Files

    train.csv - historical data including Sales
    test.csv - historical data excluding Sales
    sample_submission.csv - a sample submission file in the correct format
    store.csv - supplemental information about the stores

#### Data fields
Most of the fields are self-explanatory. The following are descriptions for those that aren't.

    Id - an Id that represents a (Store, Date) duple within the test set
    Store - a unique Id for each store
    Sales - the turnover for any given day (this is what you are predicting)
    Customers - the number of customers on a given day
    Open - an indicator for whether the store was open: 0 = closed, 1 = open
    StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
    SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
    StoreType - differentiates between 4 different store models: a, b, c, d
    Assortment - describes an assortment level: a = basic, b = extra, c = extended
    CompetitionDistance - distance in meters to the nearest competitor store
    CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
    Promo - indicates whether a store is running a promo on that day
    Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
    Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
    PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

train.csv - column name: Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday
test.csv - column name: Id, Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday
store.csv - column name: Store, StoreType, Assortment, CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2, Promo2SinceWeek, Promo2SinceYear, PromoInterval


## Dataset folder Location: 
../../kaggle-data/rossmann-store-sales. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv, store.csv

## Solution Description:
#### Features Selection / Extraction

For feature extraction, I distinguish three types of features, on 1) recent data 2) temporal information and 3) current trends. I extracted a lot more features then I ended up using.

Recent data

To create features on recent data, I selected store speciﬁc sets of sales data for each month in the the train set (i.e. the three years of sales history). Then for each record, I took the date of that record, and used data from the previous month and further back as the recent history of that record. I extracted features on last quarter, last half year, last year and last 2 years. I also experimented with last month only, but thought it would not be useful to predict sales as far as six weeks ahead.

The recent data on store as a whole were still wildly varying, and I identiﬁed three features that contributed most to this variance: the day of the week, promotions and holidays. Therefore, I further split out my store speciﬁc sets by those three variables and calculated recent averages on diverse combinations of them.

To summarize the recent data, I used measures of centrality: median, mean and harmonic mean - and measures of spread: standard deviation, skewness, kurtosis and 10%/ 90% percentiles. I also tried to log transform the sales before summarizing, but only one of those transformed variables survived the feature selection.

In one variation to the main model, I calculated the recent data features on number of customers, instead of sales amount.

Temporal information

For temporal information I created ‘day counters’ to express how each record relates to certain events or cycles. The day counter indicates either the number of days before, after or within the event. As events I had the promotion cycle (every 14 days), the secondary promotion cycle (every three months), the summer holidays (important because they partly took place during the 6 weeks test set), store refurbishments, start of competition and start of secondary promotion cycle. And also the day of week, day of month, and day/ week/ month of year. Apart from the day counters, I also added features on the number of holidays during the current week, last week and next week.

Current trends

To estimate store sales trends, I used data sets about the last quarter and the last year, similar to the data sets for recent data features. Within each dataset, I ﬁt a store speciﬁc linear model on 1) the day number - to extrapolate the trend into the six week period 2) day of week and 3) promotions. As a linear model I used Ridge regression from scikits-learn [2] with default regularization parameter, I did not try any alternatives. For each store I also calculated the year over year trend for the previous month, but that feature seems to be of minor importance.

Other information

Other important information was about the store: the dataset variables store id, assortment and storetype, together with some aggregates by store: the average sales per customer, the ratio of sales during promotions/ holidays/ Saturdays, the proportion school holidays and the proportion of days that the store is open. Finally, it helped to include data about the state speciﬁc weather: maximum temperature an mm precipitation, downloaded from the forum [3].

Selection of features and Model ensembling 

I created a lot more features - especially temporal information - than the model could handle. With all features together, the model easily overﬁt the train set, resulting in suboptimal performance on the test set. Therefore, I needed a way to reduce the feature set and select features that are most helpful to forecast into the test set. As a proxy for the test set, I used my validation set consisting of the last 6 weeks of the train set. I started off by handpicking some combinations that seemed to make sense to me. This way I soon noted that the spread features made overﬁtting easy. The
handpicking cost me a lot of time and I realized that it was biased by my ideas. Therefore I decided to create some models on random selections of the features. Some of the best performing random models gave nice improvements when I ensembled them with my handpicked models. After noting that, I ran over 500 random models and systematically calculated the validation error on each pairensemble of models. From the best model pairs I built a larger ensemble, consisting of more than 10 different models (actually the same models with different features). Then I got the idea to take the features from all of the selected models together and combine them into one model: this turned out to work very well - in the end I only kept this combined model together with two of the handpicked models. In this process as a whole, model ensembling and feature selection went hand in hand.

#### Modeling Techniques and Training

For modeling I completely relied on XGBoost and my focus was entirely on feature extraction and selection. I started off from the parameters in the (in)famous public script [4] that performed a lot better on the public leaderboard than I could reproduce within my holdout set. During some experiments, I only changed the number of rounds (from 3000 to 5000) and de column sample by tree (from 0.7 to 0.3 because I had a lot of features).

To further enrich and stabilize my ensemble of models, I added separate models that were only trained on the months May to September, over all three years. I chose these months because they cover the test set months and have some 2015 history before the test set. Another ensembling trick was to add ‘month ahead models’ that skipped the most recent month for the calculation of recent/trend features. These ‘month ahead models’ were almost as good as the models that did use all the most recent data. For prediction of the sales in the second month of the test set (where sales in the most recent month were not available at all) I used an ensemble of only ‘month ahead models’.

Like most other teams, I log transformed the dependent variable (sales) and did not include zero sales for training the model. For ﬁnal ensembling I applied the harmonic mean. I also multiplied all my predictions with a constant factor to improve them. A factor of 0.995 was optimal for both my validation set and the public test set. But I should have applied the mathematical estimate of 0.985 on the forum [5] to achieve a private leaderboard score well below 0.10.

I did compare some other models to XGBoost, but none of them were useful, neither on their own nor in the ensemble. Most of the individual XGBoost models already achieve around 0.105 on the private leaderboard - those individual models are simpler in the sense that no meta model ensembling is involved.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: