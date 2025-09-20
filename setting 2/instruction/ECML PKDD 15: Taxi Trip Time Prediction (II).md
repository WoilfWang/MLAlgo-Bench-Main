You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named ECML_PKDD_15:_Taxi_Trip_Time_Prediction_(II)_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
This is the second of two data science challenges that share the same dataset. The Taxi Service Trajectory competition predicts the final destination of taxi trips. 

To improve the efficiency of electronic taxi dispatching systems it is important to be able to predict how long a driver will have his taxi occupied. If a dispatcher knew approximately when a taxi driver would be ending their current ride, they would be better able to identify which driver to assign to each pickup request. 
In this challenge, we ask you to build a predictive framework that is able to infer the trip time of taxi rides in Porto, Portugal based on their (initial) partial trajectories. The output of such a framework must be the travel time of a particular taxi trip.
This competition is affiliated with the organization of ECML/PKDD 2015.

##  Evaluation Metric:
Submissions are evaluated one the Root Mean Squared Logarithmic Error (RMSLE). The RMSLE is calculated as

$$\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$$
Where:

    \\(n\\) is the number of hours in the test set
    \\(p_i\\) is your predicted count
    \\(a_i\\) is the actual count
    \\(\log(x)\\) is the natural logarithm

Submission Format

For every trip in the dataset, submission files should contain two columns: TRIP_ID and TRAVEL_TIME. TRIP_ID represents the ID of the trip for which you are predicting the total travel time (i.e. a string), while the TRAVEL_TIME column contains your prediction (i.e. a positive integer value containing the travel time in seconds).
The file should contain a header and have the following format:

    TRIP_ID,TRAVEL_TIME
    T1,60
    T2,90
    T3,122
    etc.

##  Dataset Description:

I. Training Dataset
We have provided an accurate dataset describing a complete year (from 01/07/2013 to 30/06/2014) of the trajectories for all the 442 taxis running in the city of Porto, in Portugal (i.e. one CSV file named "train.csv"). These taxis operate through a taxi dispatch central, using mobile data terminals installed in the vehicles. We categorize each ride into three categories: A) taxi central based, B) stand-based or C) non-taxi central based. For the first, we provide an anonymized id, when such information is available from the telephone call. The last two categories refer to services that were demanded directly to the taxi drivers on a B) taxi stand or on a C) random street.
Each data sample corresponds to one completed trip. It contains a total of 9 (nine) features, described as follows:

TRIP_ID: (String) It contains an unique identifier for each trip;
CALL_TYPE: (char) It identifies the way used to demand this service. It may contain one of three possible values:

‘A’ if this trip was dispatched from the central;
‘B’ if this trip was demanded directly to a taxi driver on a specific stand;
‘C’ otherwise (i.e. a trip demanded on a random street).

ORIGIN_CALL: (integer) It contains an unique identifier for each phone number which was used to demand, at least, one service. It identifies the trip’s customer if CALL_TYPE=’A’. Otherwise, it assumes a NULL value;
ORIGIN_STAND: (integer): It contains an unique identifier for the taxi stand. It identifies the starting point of the trip if CALL_TYPE=’B’. Otherwise, it assumes a NULL value;
TAXI_ID: (integer): It contains an unique identifier for the taxi driver that performed each trip;
TIMESTAMP: (integer) Unix Timestamp (in seconds). It identifies the trip’s start; 
DAYTYPE: (char) It identifies the daytype of the trip’s start. It assumes one of three possible values:

‘B’ if this trip started on a holiday or any other special day (i.e. extending holidays, floating holidays, etc.);
‘C’ if the trip started on a day before a type-B day;
‘A’ otherwise (i.e. a normal day, workday or weekend).

MISSING_DATA: (Boolean) It is FALSE when the GPS data stream is complete and TRUE whenever one (or more) locations are missing
POLYLINE: (String): It contains a list of GPS coordinates (i.e. WGS84 format) mapped as a string. The beginning and the end of the string are identified with brackets (i.e. [ and ], respectively). Each pair of coordinates is also identified by the same brackets as [LONGITUDE, LATITUDE]. This list contains one pair of coordinates for each 15 seconds of trip. The last list item corresponds to the trip’s destination while the first one represents its start;

The total travel time of the trip (the prediction target of this competition) is defined as the (number of points-1) x 15 seconds. For example, a trip with 101 data points in POLYLINE has a length of (101-1) * 15 = 1500 seconds. Some trips have missing data points in POLYLINE, indicated by MISSING_DATA column, and it is part of the challenge how you utilize this knowledge. 
II. Testing
Five test sets will be available to evaluate your predictive framework (in one single CSV file named "test.csv"). Each one of these datasets refer to trips that occurred between 01/07/2014 and 31/12/2014. Each one of these data sets will provide a snapshot of the current network status on a given timestamp. It will provide partial trajectories for each one of the on-going trips during that specific moment.
The five snapshots included on the test set refer to the following timestamps:
14/08/2014 18:00:0030/09/2014 08:30:0006/10/2014 17:45:0001/11/2014 04:00:0021/12/2014 14:30:00
III. Sample Submission Files
File sampleSubmission.csv uses the average travel time of all trips in the training set.
IV. Other Files
Along with these two files, we have also provided two additional files. One contains meta data regarding the taxi stands metaData_taxistandsID_name_GPSlocation.csv including id and location.
The second one includes an evaluation script for both problems developed in the R language ("evaluation_script.r").

train.csv - column name: TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND, TAXI_ID, TIMESTAMP, DAY_TYPE, MISSING_DATA, POLYLINE
metaData_taxistandsID_name_GPSlocation.csv - column name: ID, Descricao, Latitude, Longitude
test.csv - column name: TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND, TAXI_ID, TIMESTAMP, DAY_TYPE, MISSING_DATA, POLYLINE


## Dataset folder Location: 
../../kaggle-data/pkdd-15-taxi-trip-time-prediction-ii. In this folder, there are the following files you can use: train.csv, metaData_taxistandsID_name_GPSlocation.csv, sampleSubmission.csv, test.csv

## Solution Description:
One of the first observations I made was that the position and also the remaining driving time depends very strongly on the cut-off position of the trip. For certain positions it was very clear where the taxi trip is heading to (for example T15 in Fig1), and making a precise prediction is feasible. For other trips only some parts of the city could be excluded. But looking at all trips in the training set it is possible to identify these parts. This can be done by collecting all trips which were close to the cut-off position of the test trip. Figure 1 shows the start (green) and end (red) position of the collected trips for four trips in the test set. The blue dot shows the cut-off position of the corresponding test trip. We can see that for the trips T25, T50, and T59 the latitude coordinate has a strong influence on the distribution of the end points and thus also for the remaining driving time (see Fig. 1 on the right).

So I decided to train an individual model for every trip in the test dataset and to fall back to the more general ones, if the size of training data set for these models is too low. The following models were trained:

- Base model: Based on a dataset, where the features were extracted from all the tracks in the training set, and longer tracks were sampled more frequently than shorter ones.

- General expert models for short trips (e.g. 1, 2 or 3 positions are known).

- Expert models for each test trip (e.g. trained on tracks which cross the test trip at the cut-off position).

#### Pre-processing and feature engineering
The training set contained a lot of very short trips as can be seen in the left plot of Figure 2, which shows a histogram of the total trip length. The high fraction of trips less than 4 does not follow the general type of the distribution. I therefore excluded them from the analysis. Another type of error I observed was misread GPS coordinates, and I excluded them by cutting of very long distance trips. The threshold was set to 99 percent trip length for the end position predictions, and 99.9 percent for the travel time prediction, respectively.

The remaining trips where then used for the generation of the model specific training sets. Only the training set of the base model contained all trips. For this dataset every trip was randomly cut-off in between. Because the test set was collected at specific time points, longer trips are more likely included. Fig. 2 shows that by increasing the sampling frequency for longer tracks linearly with trip length, the frequency of short trips can be reduced. The resulting distribution is closer to the one of the test set (black dotted line).

The training set contained the following features:

- working day (Mon-Sun) -
- the hour the trip started 
- trip length 
- latitude coordinate start point 
- longitude coordinate start point
- latitude coordinate cut-off point
- longitude coordinate cut-off point
- distance from start point to city center
- heading towards the city center (from the start point)
- distance from city center to cut-off point
- heading from the city center (towards the cut-off point)
- net distance to cut-off point
- median velocity of the car along the track
- velocity of the car at the cut-off point
- heading of the car at the cut-off point

Interestingly, most of the meta data seemed to have little to no predictive power, so in the end I only used the time stamp.

#### Training
All models are trained using a 5 fold cross-validation technique. I used RandomForestRegressor (RFR) and GrandientBoostingRegressor (GBR) from sklearn with default settings except the number of trees, which was set to 200.

The expert models for the cut-off position were trained using both RandomForestRegressor and GradientboostingRegressor. The training set size varied from a few trips up to 100000. Models were trained only for those test trips where the training set size was above 1000. Interestingly, with increasing training set size the CV prediction error of the GradientBoostingRegressor was slightly better than the one of the RandomForestRegressor as the right plot in Fig. 3 shows.

#### Model averaging
In total I trained over 300 models, so I had to select which models to use in the final submission. Since you are allowed to select two final submissions for evaluation, I decide to use the expert models based on the cut-off position in only one of it.

Submission 1 was a blend of the base model with the expert models for short trips. Fig 3 shows that the CV error of the short trip expert model based on 3 positions is lower than the base model up to the trip length of 9. Therefore I decided to use also the short trip models for trips up to a length to 10 and 15, respectively. I calculated the average of all 4 models with equal weight and tested the submission on the public leaderboard. The blended submission with a threshold of 15 performed a little better and I used this value for the final submission. I skipped a more thorough analysis via a solid CV because of time reasons.

Submission 2 was based on submission 1. In addition the predictions were replaced by the ones of the trip expert models for all trips with sufficient training set size. I tested two thresholds (1000 and 2000) on the public leaderboard, and used 1000 for submission 2 because of the slightly better score. Again, I skipped a more thorough analysis via a solid CV because of time reasons.

Interestingly, submission 1 scored better in the private leaderboard than submission 2 (0.5092 vs. 0.5045), whereas for the public leaderboard it was the other way around (0.5253 vs. 0.5354).



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: