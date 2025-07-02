You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_Mar_2022_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
For the March edition of the 2022 Tabular Playground Series you're challenged to forecast twelve-hours of traffic flow in a U.S. metropolis. The time series in this dataset are labelled with both location coordinates and a direction of travel -- a combination of features that will test your skill at spatio-temporal forecasting within a highly dynamic traffic network.

Which model will prevail? The venerable linear regression? The deservedly-popular ensemble of decision trees? Or maybe a cutting-edge graph neural-network? We can't wait to see!

About the Tabular Playground Series

Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly.

The goal of these competitions is to provide a fun and approachable-for-anyone tabular dataset to model. These competitions are a great choice for people looking for something in between the Titanic Getting Started competition and the Featured competitions. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you; thus, we encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals.

##  Evaluation Metric:
Submissions are evaluated on the mean absolute error between predicted and actual congestion values for each time period in the test set.

Submission File

For each row_id in the test set, you should predict a congestion measurement. The file should contain a header and have the following format:

    row_id,congestion
    140140,0.0
    140141,0.0
    140142,0.0
    ...

The congestion target has integer values from 0 to 100, but your predictions may be any floating-point number.

##  Dataset Description:
In this competition, you'll forecast twelve-hours of traffic flow in a major U.S. metropolitan area. Time, space, and directional features give you the chance to model interactions across a network of roadways.

Files and Field Descriptions

train.csv - the training set, comprising measurements of traffic congestion across 65 roadways from April through September of 1991.

    row_id - a unique identifier for this instance
    time - the 20-minute period in which each measurement was taken
    x - the east-west midpoint coordinate of the roadway
    y - the north-south midpoint coordinate of the roadway
    direction - the direction of travel of the roadway. EB indicates "eastbound" travel, for example, while SW indicates a "southwest" direction of travel.
    congestion - congestion levels for the roadway during each hour; the target. The congestion measurements have been normalized to the range 0 to 100.

test.csv - the test set; you will make hourly predictions for roadways identified by a coordinate location and a direction of travel on the day of 1991-09-30.

sample_submission.csv - a sample submission file in the correct format

Source
This dataset was derived from the Chicago Traffic Tracker - Historical Congestion Estimates dataset.

train.csv - column name: row_id, time, x, y, direction, congestion
test.csv - column name: row_id, time, x, y, direction


## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-mar-2022. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
When I first saw that TPS-22-03 is rather small timeseries, I was looking forward to diving deep in all those autocorrelations and crosscorrelations. However during EDA it was pretty soon noticable that this data does not behave as timeseries (it was synthetic dataset, right?). I have created 4 baseline submissions and left this topic for almost 20 days.

After my return I briefly skimmed through some public notebooks (I cannot recall which, sorry) and took away two possible upgrades:

    it's beneficial to have categories even with time (I previously created categories only with x-y-dir),
    some sort of expanding window.

I created features with expanding mean with window of 7 days for each x-y-dir-hour and x-y-dir-hour_minute, similarly rolling mean with window of 5 last days for specific x-y-dir-hour and expanding mean of rolling-mean-smoothed congestion. To prevent data leakage all these features were lagged by 7 days for corresponding x-y-dir.

As final model LightGBM regressor was used - only notable thing is that training was done only on workdays. Predictions were rounded to integers.

Last night I was hoping to break into Top-200, jump straight to podium is from other world. But I am aware that as the prediction window was only 12 hours of a single day, the final result is likely luck-based.

However I'm still pleasantly surprised and already looking forward to next dataset.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: