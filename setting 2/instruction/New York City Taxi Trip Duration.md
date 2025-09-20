You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named New_York_City_Taxi_Trip_Duration_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
In this competition, Kaggle is challenging you to build a model that predicts the total ride duration of taxi trips in New York City. Your primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables.
Longtime Kagglers will recognize that this competition objective is similar to the ECML/PKDD trip time challenge we hosted in 2015. But, this challenge comes with a twist. Instead of awarding prizes to the top finishers on the leaderboard, this playground competition was created to reward collaboration and collective learning. 
We are encouraging you (with cash prizes!) to publish additional training data that other participants can use for their predictions. We also have designated bi-weekly and final prizes to reward authors of kernels that are particularly insightful or valuable to the community.

##  Evaluation Metric:
The evaluation metric for this competition is Root Mean Squared Logarithmic Error.
The RMSLE is calculated as
$$\epsilon = \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$$
Where:
\\(\epsilon\\) is the RMSLE value (score)\\(n\\) is the total number of observations in the (public/private) data set,\\(p_i\\) is your prediction of trip duration, and\\(a_i\\) is the actual trip duration for \\(i\\). \\(\log(x)\\) is the natural logarithm of \\(x\\)

##### Submission File
For every row in the dataset, submission files should contain two columns: id and trip_duration.  The id corresponds to the column of that id in the test.csv. The file should contain a header and have the following format:

    id,trip_duration 
    id00001,978
    id00002,978
    id00003,978
    id00004,978
    etc.

##  Dataset Description:
The competition dataset is based on the 2016 NYC Yellow Cab trip record data made available in Big Query on Google Cloud Platform. The data was originally published by the NYC Taxi and Limousine Commission (TLC). The data was sampled and cleaned for the purposes of this playground competition. Based on individual trip attributes, participants should predict the duration of each trip in the test set.
File descriptions

    train.csv - the training set (contains 1458644 trip records)
    test.csv - the testing set (contains 625134 trip records)
    sample_submission.csv - a sample submission file in the correct format

Data fields

    id - a unique identifier for each trip
    vendor_id - a code indicating the provider associated with the trip record
    pickup_datetime - date and time when the meter was engaged
    dropoff_datetime - date and time when the meter was disengaged
    passenger_count - the number of passengers in the vehicle (driver entered value)
    pickup_longitude - the longitude where the meter was engaged
    pickup_latitude - the latitude where the meter was engaged
    dropoff_longitude - the longitude where the meter was disengaged
    dropoff_latitude - the latitude where the meter was disengaged
    store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
    trip_duration - duration of the trip in seconds

Disclaimer: The decision was made to not remove dropoff coordinates from the dataset order to provide an expanded set of variables to use in Kernels.

train.csv - column name: id, vendor_id, pickup_datetime, dropoff_datetime, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, store_and_fwd_flag, trip_duration
test.csv - column name: id, vendor_id, pickup_datetime, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, store_and_fwd_flag


## Dataset folder Location: 
../../kaggle-data/nyc-taxi-trip-duration. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Hi, all
Thanks to the hoster and all Kagglers that contributed to the discussion and kernels. Because of the data leak, this rank is not precise. But still, I'd like to share my solutions.

#### Features:
Thanks to beluga's sharing. Most of my features are based on his strategy.
Apart of this part, I also clustered the center coordinate of the trip and created similar aggregated features such as average speed and count of center cluster based on time series.
I discretized the feature of direction into 4 bins (southwest, southeast, northwest, northeast) and 8 bins.
For the date_time feature, I only created a new feature called is_rest_day which assigns 1 to weekends and holidays, 0 to business days.
A new average speed feature using total_travel_time in fastest_route data set divided by trip_duration.
More aggregated features (average speed and count) are created through grouping date_time features(hour, week_hour, is_rest_day), cluster features(pick_up_cluster, drop_off_cluster, center_cluster), vendor_id and discretized directions. I totally have 18 different groups of aggregated features. Since there are lots of correlations among them, I divided them into 4 different data sets.
I rotated the latitude and longitude to make the Manhattan road direction correspond with the new direction. For example, the whole 5th Ave will be on the same longitude. Then I calculated the trip distance on new North-South direction and East-West direction respectively. Two new direction feature: (North or South) and (East or West). I also clustered the new coordinate and  generated some aggregated features.
For the fastest_route data set, I applied three solutions. 

1. Firstly, I created a dict for each trip of their routes and relative distances, for example: {'5th Ave': 500, 'W 26th St': 180}  (Just an assumption, not real number). Then I used DictVectorizor to create a sparse matrix. The final step is applying SVD on this sparse matrix and using the top 50 features which explained about 87% of the variance.
2. I aggregated routes frequency by hour and week_hour. For example, W 26th St occurs 200 times in Wednesday hour 16. Such frequency may represent the traffic situation of roads within specific time. Then, for each trip, I calculated their mean and sum frequency for the routes it went through. 
3. Following the second solution. I multiplied the frequencies by route distances of each trip, created new dictsm transformed into a new sparse matrix and applied SVD.

Model:
A simple 2-layer stacking. 
1st layer: Since I have 4 different aggregated data sets. For each data set, I applied a XGB, a RandomForest, a ExtraTrees, a DecisionTree and a Linear Regression. Totally 20 models
2nd layer, just a carefully tuned XGB.
That's all what I did. Thanks!.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: