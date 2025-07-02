You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named M5_Forecasting_-_Accuracy_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Note: This is one of the two complementary competitions that together comprise the M5 forecasting challenge. Can you estimate, as precisely as possible, the point forecasts of the unit sales of various products sold in the USA by Walmart? If you are interested in estimating the uncertainty distribution of the realized values of the same series, be sure to check out its companion competition

How much camping gear will one store sell each month in a year? To the uninitiated, calculating sales at this level may seem as difficult as predicting the weather. Both types of forecasting rely on science and historical data. While a wrong weather forecast may result in you carrying around an umbrella on a sunny day, inaccurate business forecasts could result in actual or opportunity losses.  In this competition, in addition to traditional forecasting methods you’re also challenged to use machine learning to improve forecast accuracy.

The Makridakis Open Forecasting Center (MOFC) at the University of Nicosia conducts cutting-edge forecasting research and provides business forecast training. It helps companies achieve accurate predictions, estimate the levels of uncertainty, avoiding costly mistakes, and apply best forecasting practices. The MOFC is well known for its Makridakis Competitions, the first of which ran in the 1980s.

In this competition, the fifth iteration, you will use hierarchical sales data from Walmart, the world’s largest company by revenue, to forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.

If successful, your work will continue to advance the theory and practice of forecasting. The methods used can be applied in various business areas, such as setting up appropriate inventory or service levels. Through its business support and training, the MOFC will help distribute the tools and knowledge so others can achieve more accurate and better calibrated forecasts, reduce waste and be able to appreciate uncertainty and its risk implications.


##  Evaluation Metric:
This competition uses a Weighted Root Mean Squared Scaled Error (RMSSE). Extensive details about the metric, scaling, and weighting can be found in the M5 Participants Guide.

Submission File

Each row contains an id that is a concatenation of an item_id and a store_id, which is either validation (corresponding to the Public leaderboard), or evaluation (corresponding to the Private leaderboard). You are predicting 28 forecast days (F1-F28) of items sold for each row. For the validation rows, this corresponds to d_1914 - d_1941, and for the evaluation rows, this corresponds to d_1942 - d_1969. (Note: a month before the competition close, the ground truth for the validation rows will be provided.)

The files must have a header and should look like the following:

    id,F1,...F28
    HOBBIES_1_001_CA_1_validation,0,...,2
    HOBBIES_1_002_CA_1_validation,2,...,11
    ...
    HOBBIES_1_001_CA_1_evaluation,3,...,7
    HOBBIES_1_002_CA_1_evaluation,1,...,4


##  Dataset Description:
In the challenge, you are predicting item sales at stores in various locations for two 28-day time periods. Information about the data is found in the M5 Participants Guide.

Files

    calendar.csv - Contains information about the dates on which the products are sold.
    sales_train_validation.csv - Contains the historical daily unit sales data per product and store [d_1 - d_1913]
    sample_submission.csv - The correct format for submissions. Reference the Evaluation tab for more info.
    sell_prices.csv - Contains information about the price of the products sold per store and date.
    sales_train_evaluation.csv - Includes sales [d_1 - d_1941] (labels used for the Public leaderboard)

sell_prices.csv - column name: store_id, item_id, wm_yr_wk, sell_price
calendar.csv - column name: date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type_1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI


## Dataset folder Location: 
../../kaggle-data/m5-forecasting-accuracy. In this folder, there are the following files you can use: sell_prices.csv, sample_submission.csv, sales_train_validation.csv, sales_train_evaluation.csv, calendar.csv

## Solution Description:
### Model
LightGBM (single)  
objective = tweedie

### Validation
5 holdout (d1578-d1605, d1830-d1857, d1858-d1885, d1886-d1913, d1914-d1941)  
no early stopping

### Model split
for each store

    for each week

        model w1 predicts F01, F02, …, F07
        model w2 predicts F08, F09, …, F14
        model w3 predicts F15, F16, …, F21
        model w4 predicts F22, F23, …, F28

### Features
    General time-series features
    General price features
    General calendar features
    No recursive features

When the competition began, I first understood the data and evaluation metrics and created a baseline model.
Then I realized that the validation scores varied significantly over the period of time. (I can't build a proper validation.)
I was strongly convinced that this competition would be very difficult and that it would be impossible to build a model with high accuracy.

So I decided to give up on trying to get a high ranking in the competition. Instead of it, I decided to try to build a "practical" solution.

My strategy is as follows:

    Not to use post processing, multiplier, and leaks

        In practice, it is not possible to utilize such information, so I have decided that it should not be used.
    Trust CV but Not care about official evaluation metrics (WRMSSE)

        WRMSSE is evaluation metrics by the competition organizer, but in practice, I think WRMSSE is not always reasonable. Therefore I didn't make a custom loss function to avoid overfit this competition task itself.
    Not to use complex and recursive features

        Just use general features that can be applied to any practical tasks.
        Recursive features lead to error accumulation
    Not to use computation resources too much

        Single model (LightGBM only, no stacking/blending)
        Memory efficiency (model for each store and week)

As I mentioned earlier, I was not aiming for the high ranking, so I was very surprised by this result (4th place).
As you can see from the solution above, I'm not doing anything special.

What was valuable to me was learning a lot about building a simple solution that can be widely used in practice. The ranking itself means absolutely nothing to me.

(However, I feel very sorry for those of you who put in a great deal of effort to get to the top…)

Either way, I have learned a lot from this competition and community.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: