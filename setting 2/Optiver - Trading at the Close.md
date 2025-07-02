You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Optiver_-_Trading_at_the_Close_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
In this competition, you are challenged to develop a model capable of predicting the closing price movements for hundreds of Nasdaq listed stocks using data from the order book and the closing auction of the stock. Information from the auction can be used to adjust prices, assess supply and demand dynamics, and identify trading opportunities.

Stock exchanges are fast-paced, high-stakes environments where every second counts. The intensity escalates as the trading day approaches its end, peaking in the critical final ten minutes. These moments, often characterised by heightened volatility and rapid price fluctuations, play a pivotal role in shaping the global economic narrative for the day.

Each trading day on the Nasdaq Stock Exchange concludes with the Nasdaq Closing Cross auction. This process establishes the official closing prices for securities listed on the exchange. These closing prices serve as key indicators for investors, analysts and other market participants in evaluating the performance of individual securities and the market as a whole.

Within this complex financial landscape operates Optiver, a leading global electronic market maker. Fueled by technological innovation, Optiver trades a vast array of financial instruments, such as derivatives, cash equities, ETFs, bonds, and foreign currencies, offering competitive, two-sided prices for thousands of these instruments on major exchanges worldwide.

In the last ten minutes of the Nasdaq exchange trading session, market makers like Optiver merge traditional order book data with auction book data. This ability to consolidate information from both sources is critical for providing the best prices to all market participants.

In this competition, you are challenged to develop a model capable of predicting the closing price movements for hundreds of Nasdaq listed stocks using data from the order book and the closing auction of the stock. Information from the auction can be used to adjust prices, assess supply and demand dynamics, and identify trading opportunities.

Your model can contribute to the consolidation of signals from the auction and order book, leading to improved market efficiency and accessibility, particularly during the intense final ten minutes of trading. You'll also get firsthand experience in handling real-world data science problems, similar to those faced by traders, quantitative researchers and engineers at Optiver.

##  Evaluation Metric:
Submissions are evaluated on the Mean Absolute Error (MAE) between the predicted return and the observed target. The formula is given by:
$$MAE = \frac{1}{n} \sum\limits_{i=1}^{n} {|y_i - x_i|}$$
Where:

    n  is the total number of data points.
    y_i is the predicted value for data point i.
    x_i is the observed value for data point i. 


##  Dataset Description:
This dataset contains historic data for the daily ten minute closing auction on the NASDAQ stock exchange. Your challenge is to predict the future price movements of stocks relative to the price future price movement of a synthetic index composed of NASDAQ-listed stocks.
This is a forecasting competition using the time series API. The private leaderboard will be determined using real market data gathered after the submission period closes.

#### Files
[train/test].csv The auction data. The test data will be delivered by the API.

    stock_id - A unique identifier for the stock. Not all stock IDs exist in every time bucket.
    date_id - A unique identifier for the date. Date IDs are sequential & consistent across all stocks.
    imbalance_size - The amount unmatched at the current reference price (in USD).
    imbalance_buy_sell_flag - An indicator reflecting the direction of auction imbalance.
            buy-side imbalance; 1
            sell-side imbalance; -1
            no imbalance; 0
    reference_price - The price at which paired shares are maximized, the imbalance is minimized and the distance from the bid-ask midpoint is minimized, in that order. Can also be thought of as being equal to the near price bounded between the best bid and ask price.
    matched_size - The amount that can be matched at the current reference price (in USD).
    far_price - The crossing price that will maximize the number of shares matched based on auction interest only. This calculation excludes continuous market orders.
    near_price - The crossing price that will maximize the number of shares matched based auction and continuous market orders.
    [bid/ask]_price - Price of the most competitive buy/sell level in the non-auction book.
    [bid/ask]_size - The dollar notional amount on the most competitive buy/sell level  in the non-auction book.
    wap - The weighted average price in the non-auction book. \frac{ {BidPrice * AskSize + AskPrice * BidSize}}{BidSize + AskSize}
    seconds_in_bucket - The number of seconds elapsed since the beginning of the day's closing auction, always starting from 0.
    target - The 60 second future move in the wap of the stock, less the 60 second future move of the synthetic index. Only provided for the train set.
    The synthetic index is a custom weighted index of Nasdaq-listed stocks constructed by Optiver for this competition.
    The unit of the target is basis points, which is a common unit of measurement in financial markets. A 1 basis point price move is equivalent to a 0.01% price move.
    Where t is the time at the current observation, we can define the target:
    Target = (\frac{StockWAP_{t+60}}{StockWAP_{t}} - \frac{IndexWAP_{t+60}}{IndexWAP_{t}}) * 10000

All size related columns are in USD terms.

All price related columns are converted to a price move relative to the stock wap (weighted average price) at the beginning of the auction period.

sample_submission A valid sample submission, delivered by the API. See this notebook for a very simple example of how to use the sample submission.

revealed_targets When the first time_id for each date  (i.e. when seconds_in_bucket equals zero) the API will serve a dataframe providing the true target values for the entire previous date. All other rows contain null values for the columns of interest.

public_timeseries_testing_util.py An optional file intended to make it easier to run custom offline API tests. See the script's docstring for details. You will need to edit this file before using it.

example_test_files/ Data intended to illustrate how the API functions. Includes the same files and columns delivered by the API. The first three date ids are repeats of the last three date ids in the train set, to enable an illustration of how the API functions.

optiver2023/ Files that enable the API. Expect the API to deliver all rows in under five minutes and to reserve less than 0.5 GB of memory. The first three date ids delivered by the API are repeats of the last three date ids in the train set, to better illustrate how the API functions. You must make predictions for those dates in order to advance the API but those predictions are not scored.

train.csv - column name: stock_id, date_id, seconds_in_bucket, imbalance_size, imbalance_buy_sell_flag, reference_price, matched_size, far_price, near_price, bid_price, bid_size, ask_price, ask_size, wap, target, time_id, row_id
test.csv - column name: stock_id, date_id, seconds_in_bucket, imbalance_size, imbalance_buy_sell_flag, reference_price, matched_size, far_price, near_price, bid_price, bid_size, ask_price, ask_size, wap, time_id, row_id, currently_scored


## Dataset folder Location: 
../../kaggle-data/optiver-trading-at-the-close. In this folder, there are the following files you can use: train.csv, test.csv

## Solution Description:
Thanks to Optiver and Kaggle for hosting this great financial competition. And thanks to the
great notebooks and discussions, I learned a lot. I am so happy to win my second solo win! 😃😀😀

#### Overview
My final model(CV/Private LB of 5.8117/5.4030) was a combination of CatBoost (5.8240/5.4165), GRU (5.8481/5.4259), and Transformer (5.8619/5.4296), with respective weights of 0.5, 0.3, 0.2 searched from validation set. And these models share same 300 features.
Besides, online learning(OL) and post-processing(PP) also play an important role in my final submission.


#### Validation Strategy
My validation strategy is pretty simple, train on first 400 days and choose last 81 days as my holdout validation set. The CV score aligns with leaderboard score very well which makes me believe that this competition wouldn't shake too much. So I just focus on improving CV in most of time.

#### Magic Features
My models have 300 features in the end. Most of these are commonly used, such like raw price, mid price, imbalance features, rolling features and historical target features.

I will introduce some features really helpful and other teams didn't share yet.

1 agg features based on seconds_in_bucket_group

pl.when(pl.col('seconds_in_bucket') < 300).then(0).when(pl.col('seconds_in_bucket') < 480).then(1).otherwise(2).cast(pl.Float32).alias('seconds_in_bucket_group'),

 *[(pl.col(col).first() / pl.col(col)).over(['date_id', 'seconds_in_bucket_group', 'stock_id']).cast(pl.Float32).alias('{}_group_first_ratio'.format(col)) for col in base_features],
 *[(pl.col(col).rolling_mean(100, min_periods=1) / pl.col(col)).over(['date_id', 'seconds_in_bucket_group', 'stock_id']).cast(pl.Float32).alias('{}_group_expanding_mean{}'.format(col, 100)) for col in base_features]

2 rank features grouped by seconds_in_bucket

 *[(pl.col(col).mean() / pl.col(col)).over(['date_id', 'seconds_in_bucket']).cast(pl.Float32).alias('{}_seconds_in_bucket_group_mean_ratio'.format(col)) for col in base_features],
 *[(pl.col(col).rank(descending=True,method='ordinal') / pl.col(col).count()).over(['date_id', 'seconds_in_bucket']).cast(pl.Float32).alias('{}_seconds_in_bucket_group_rank'.format(col)) for col in base_features],

#### Feature Selection
Feature selection is important because we have to avoid memory error issue and run as many rounds of online training as possible. 

I just choose top 300 features by CatBoost model's feature importance.

#### Model

1. Nothing to say about CatBoost as usual, just simply train and predict.
2. GRU input tensor's shape is (batch_size, 55 time steps,  dense_feature_dim), followed by 4 layers GRU, output tensor's shape is (batch_size, 55 time steps).
3. Transformer input tensor's shape is (batch_size, 200 stocks,  dense_feature_dim), followed by 4 layers transformer encoder layers, output tensor's shape is (batch_size, 200 stocks). A small trick that turns output into zero mean is helpful.

out = out - out.mean(1, keepdim=True)
4. set sample weight: 
(pl.col('date_id') / pl.col('date_id').max() * 10.0 + 1.0).clip(1,100).cast(pl.Float32).alias('sample_weight')

#### Online Learning Strategy
I retrain my model every 12 days, 5 times in total. 

I think most teams can only use up to 200 features when training GBDT if online training strategy is adopted. Because it requires double memory consumption when concat historical data with online data. 

The data loading trick can greatly increase this. For achieving this,  you should save training data one file per day and also loading day by day.
data loading trick

def load_numpy_data(meta_data, features):
    res = np.empty((len(meta_data), len(features)), dtype=np.float32)
    all_date_id = sorted(meta_data['date_id'].unique())
    data_index = 0
    for date_id in tqdm(all_date_id):
        tmp = h5py.File( '/path/to/{}.h5'.format(date_id), 'r')
        tmp = np.array(tmp['data']['features'], dtype=np.float32)
        res[data_index:data_index+len(tmp),:] = tmp
        data_index += len(tmp)
    return res

Actually, my best submission is overtime at last update. I just skip online training if total inference time meets certain value.

So there are 4 online training updates in total. I estimate that the best score would be around 5.400 if not overtime.
Anyway, I am really lucky! 

#### Post Processing
Subtract weighted-mean is better than average-mean since metric already told.

test_df['stock_weights'] = test_df['stock_id'].map(stock_weights)
test_df['target'] = test_df['target'] - (test_df['target'] * test_df['stock_weights']).sum() / test_df['stock_weights'].sum()



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: