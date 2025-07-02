You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Corporación_Favorita_Grocery_Sales_Forecasting_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Brick-and-mortar grocery stores are always in a delicate dance with purchasing and sales forecasting. Predict a little over, and grocers are stuck with overstocked, perishable goods. Guess a little under, and popular items quickly sell out, leaving money on the table and customers fuming.

The problem becomes more complex as retailers add new locations with unique needs, new products, ever transitioning seasonal tastes, and unpredictable product marketing. Corporación Favorita, a large Ecuadorian-based grocery retailer, knows this all too well. They operate hundreds of supermarkets, with over 200,000 different products on their shelves.

Corporación Favorita has challenged the Kaggle community to build a model that more accurately forecasts product sales. They currently rely on subjective forecasting methods with very little data to back them up and very little automation to execute plans. They’re excited to see how machine learning could better ensure they please customers by having just enough of the right products at the right time.

##  Evaluation Metric:
Submissions are evaluated on the Normalized Weighted Root Mean Squared Logarithmic Error (NWRMSLE), calculated as follows:
$$ NWRMSLE = \sqrt{ \frac{\sum_{i=1}^n w_i \left( \ln(\hat{y}i + 1) - \ln(y_i +1)  \right)^2  }{\sum{i=1}^n w_i}} $$
where for row i, \(\hat{y}_i\) is the predicted unit_sales of an item and \(y_i\) is the actual unit_sales;  n is the total number of rows in the test set.

The weights, \(w_i\), can be found in the items.csv file (see the Data page). Perishable items are given a weight of 1.25 where all other items are given a weight of 1.00.

This metric is suitable when predicting values across a large range of orders of magnitudes. It avoids penalizing large differences in prediction when both the predicted and the true number are large: predicting 5 when the true value is 50 is penalized more than predicting 500 when the true value is 545.

Submission File

For eachid in the test set, you must predict theunit_sales. Because the metric uses ln(y+1), submissions are validated to ensure there are no negative predictions. 

The file should contain a header and have the following format:

    id,unit_sales
    125497040,2.5
    125497041,0.0
    125497042,27.9
    etc.


##  Dataset Description:
In this competition, you will be predicting the unit sales for thousands of items sold at different Favorita stores located in Ecuador. The training data includes dates, store and item information, whether that item was being promoted, as well as the unit sales. Additional files include supplementary information that may be useful in building your models.
#### File Descriptions and Data Field Information

train.csv

    Training data, which includes the target unit_sales by date, store_nbr, and item_nbr and a unique id to label rows.
    The target unit_sales can be integer (e.g., a bag of chips) or float (e.g., 1.5 kg of cheese).
    Negative values of unit_sales represent returns of that particular item.
    The onpromotion column tells whether that item_nbr was on promotion for a specified date and store_nbr.
    Approximately 16% of the onpromotion values in this file are NaN.
    NOTE: The training data does not include rows for items that had zero unit_sales for a store/date combination. There is no information as to whether or not the item was in stock for the store on the date, and teams will need to decide the best way to handle that situation. Also, there are a small number of items seen in the training data that aren't seen in the test data.

test.csv

    Test data, with the date, store_nbr, item_nbr combinations that are to be predicted, along with the onpromotion information.
    NOTE: The test data has a small number of items that are not contained in the training data. Part of the exercise will be to predict a new item sales based on similar products..
    The public / private leaderboard split is based on time. All items in the public split are also included in the private split.

sample_submission.csv

    A sample submission file in the correct format.
    It is highly recommend you zip your submission file before uploading!

stores.csv

    Store metadata, including city, state, type, and cluster.
    cluster is a grouping of similar stores.

items.csv

    Item metadata, including family, class, and perishable.
    NOTE: Items marked as perishable have a score weight of 1.25; otherwise, the weight is 1.0.

transactions.csv

    The count of sales transactions for each date, store_nbr combination. Only included for the training data timeframe.

oil.csv

    Daily oil price. Includes values during both the train and test data timeframe. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)

holidays_events.csv

    Holidays and Events, with metadata
    NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
    Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).

Additional Notes

    Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
    A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.

holidays_events.csv - column name: date, type, locale, locale_name, description, transferred
train.csv - column name: id, date, store_nbr, item_nbr, unit_sales, onpromotion
stores.csv - column name: store_nbr, city, state, type, cluster
transactions.csv - column name: date, store_nbr, transactions
items.csv - column name: item_nbr, family, class, perishable
test.csv - column name: id, date, store_nbr, item_nbr, onpromotion
oil.csv - column name: date, dcoilwtico


## Dataset folder Location: 
../../kaggle-data/favorita-grocery-sales-forecasting. In this folder, there are the following files you can use: holidays_events.csv, train.csv, stores.csv, transactions.csv, items.csv, test.csv, sample_submission.csv, oil.csv

## Solution Description:
#### Model Overview
I build 3 models: a lightGBM, a CNN+DNN and a seq2seq RNN model. Final model was a weighted average of these models (where each model is stabilized by training multiple times with different random seeds then take the average), with special dealing with promotion, which I will discuss later. Each model separately can stay in top 1% in the private leaderboard.

LGBM: It is an upgraded version of the public kernels. More features, data and periods were fed to the model.

CNN+DNN: This is a traditional NN model, where the CNN part is a dilated causal convolution inspired by WaveNet, and the DNN part is 2 FC layers connected to raw sales sequences. Then the inputs are concatenated together with categorical embeddings and future promotions, and directly output to 16 future days of predictions. 

The detailed architecture is:
Inputs:
	1.	Sequential Data Inputs:
	•	seq_in: Sequence of numerical values with shape (timesteps, 1).
	•	is0_in: Additional sequential numerical data with shape (timesteps, 1).
	•	promo_in, yearAgo_in, quarterAgo_in: Promotional and temporal features with shape (timesteps+16, 1).
	•	item_mean_in, store_mean_in: Mean statistics for items and stores with shape (timesteps, 1).
	2.	Categorical Inputs:
	•	weekday_in: Day of the week encoded as integers with shape (timesteps+16,).
	•	dom_in: Day of the month encoded as integers with shape (timesteps+16,).
	•	cat_features: A set of six categorical features related to items and stores with shape (6,).

Embedding Layers:
	•	Weekday and Day of Month:
	•	weekday_embed_encode: Embeds weekday information into a 4-dimensional space.
	•	dom_embed_encode: Embeds day-of-month information into a 4-dimensional space.
	•	Categorical Features:
	•	family_embed: Embeds item family into an 8-dimensional space.
	•	store_embed: Embeds store number into an 8-dimensional space.
	•	cluster_embed: Embeds store cluster into a 3-dimensional space.
	•	type_embed: Embeds store type into a 2-dimensional space.

Convolutional Pathway:
	1.	Concatenation:
	•	Combines seq_in, a sliced portion of promo_in, and item_mean_in along the feature axis.
	2.	Dilated Convolutions:
	•	First Layer (c1): A Conv1D layer with 32 filters, kernel size 2, dilation rate 1, causal padding, and ReLU activation.
	•	Subsequent Layers (c2): Three stacked Conv1D layers with the same number of filters and kernel size but increasing dilation rates (2, 4, and 8 respectively) to capture wider temporal dependencies.
	3.	Combination and Processing:
	•	Concatenates outputs from c1 and c2.
	•	Applies a Conv1D layer with 8 filters and kernel size 1, followed by a Dropout layer (25%) and flattening to produce a feature vector.

Dense (Fully Connected) Pathway:
	1.	Processing Sequential Input:
	•	Flattens seq_in and passes it through two Dense layers with 512 and 256 neurons respectively, each followed by ReLU activation.
	•	Applies a Dropout layer (25%) to reduce overfitting.
	2.	Feature Integration:
	•	Concatenates the convolutional features, dense pathway features, flattened promotional predictions, embedded categorical features (family_embed, store_embed, cluster_embed, type_embed), and item_perish.
	3.	Final Dense Layers:
	•	Passes the concatenated features through another Dense layer with 512 neurons and ReLU activation, followed by a Dropout layer (25%).
	•	Outputs the final prediction through a Dense layer with 16 neurons and ReLU activation.

The network effectively combines convolutional layers with varying dilation rates to capture temporal patterns in the sequential data while integrating rich categorical information through embedding layers. 

RNN: This is a seq2seq model with a similar architecture of @Arthur Suilin's solution for the web traffic prediction. Encoder and decoder are both GRUs. The hidden states of the encoder are passed to the decoder through an FC layer connector. This is useful to improve the accuracy significantly.

Feature Engineering

For LGB, for each time periods the mean sales, count of promotions and count of zeros are included. These features are calculated with different ways of splits of the time periods, e.g. with/without promotion, each weekdays, item/store group stat, etc. Categorical features are included with label encoding.

For NN, item mean and year-ago/quarter-ago sales are fed as sequences input. Categorical features and time features (weekday, day of month) are fed as embeddings

Training and Validation

For training and validation, only store-item combinations that are appeared in the test data and have at least one record during 2017 in the training data are included. Validation period is 2017.7.26~2017.8.10 throughout the competition, and training periods are collected randomly during 2017.1.1~2017.7.5.

For LGB the number of range is 18, total # of data is ~3M. For NN the data is randomly generated batch by batch. Batch size is 1500 and steps per epoch is 1500 so total # of data trained per epoch is ~2M.

onPromotion

As you guys all noticed, there are some problems with the promotion feature here. Besides the missing promotion information for zero sales in training data, I found that the promotion info in 8/16 is quite abnormal and unreliable (e.g. there are ~6000 unseen store-item combo on promotion on 8/16). So I don't think it reliable to use PB to infer the distribution of promotion in test set.

The sales ratio of promo/non-promo items in training set is ~2:1 if I fill missing values with 0. I try to infer this ratio in test data. We know that the proportion of 1s in training promo data is underestimated because I fill all missing values with 0. the proportion of 1s in training set is ~6.3%, while in test set is ~6.9% without 8/16. The proportion of missing value is ~40% in training data. If I assume the true distribution is consistent, the proportion of items that are missing in the training data but actually has promotion is (0.069-0.063)/0.4 = ~1.5%. And as the missing values are all 0s, the true ratio would be (0.0632 + (0.069-0.063)0)/0.069 = ~1.83. So I guess our training model is around 10% over-estimating the items on promotion.

Then I simple re-train all the models, but without the promotion information. These predictions will of course have a lower sales ratio between promo/non-promo items. Then I average the no-promo predictions with the original predictions with promotion info with weights so that the sales ratio of the final predictions approaches 1.83:1 as I inferred.

This approach is kind of tricky and based on assumptions I cannot validated, but that's the only way I figure out to deal with the promotion bias. However, it seems like it's not useful to the PB. Ironically, the model without any special dealing with promotion bias gives me .510 on PB. I still have no idea why.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: