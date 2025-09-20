You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_Jul_2021_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly. 

In order to have a more consistent offering of these competitions for our community, we're trying a new experiment in 2021. We'll be launching month-long tabular Playground competitions on the 1st of every month and continue the experiment as long as there's sufficient interest and participation.

The goal of these competitions is to provide a fun, and approachable for anyone, tabular dataset. These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you. We encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals. 

##  Evaluation Metric:
Submissions are evaluated using the mean column-wise root mean squared  logarithmic error.
The RMSLE for a single column calculated as:
$$\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 },$$
where:

    n is the total number of observations
    pi is your prediction
    ai is the actual value
    log(x) is the natural logarithm of x
The final score is the mean of the RMSLE over all columns, in this case, 3.

Submission File
For each ID in the test set, you must predict a probability for the TARGET variable. The file should contain a header and have the following format:

    date_time,target_carbon_monoxide,target_benzene,target_nitrogen_oxides
    2011-01-01 01:00:00,2.0,10.0,300.0
    2011-01-01 02:00:00,2.0,10.0,300.0
    2011-01-01 03:00:00,2.0,10.0,300.0
    etc.


##  Dataset Description:
In this competition you are predicting the values of air pollution measurements over time, based on basic weather information (temperature and humidity) and the input values of 5 sensors.

The three target values to you to predict are: target_carbon_monoxide, target_benzene, and target_nitrogen_oxides

Files

    train.csv - the training data, including the weather data, sensor data, and values for the 3 targets
    test.csv - the same format as train.csv, but without the target value; your task is to predict the value for each of these targets.
    sample_submission.csv - a sample submission file in the correct format.

train.csv - column name: date_time, deg_C, relative_humidity, absolute_humidity, sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, target_carbon_monoxide, target_benzene, target_nitrogen_oxides
test.csv - column name: date_time, deg_C, relative_humidity, absolute_humidity, sensor_1, sensor_2, sensor_3, sensor_4, sensor_5


## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-jul-2021. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Hi everyone!
I had loads of fun playing and reading many of your discussions and codes! Below is a description of what I used and some of my thoughts. I learned a lot thanks to all of you! However, I need to publicly thank Alexander Ryzhkov @alexryzhkov for his very helpful discussions on Pseudo-Labeling and for publicly posting one of his notebooks. He deserves credit!

Feature engineering: Weirdly enough, I ended up not including any new variables beyond what was provided (+ the leaked data). I tried adding a wide range of variables, one at a time (dummies for months, weekdays, hours, etc. combinations of the weather variables, several interactions, lags, etc. etc.) but each of them worsened my score. This may have prevented overfitting the private dataset and hence may have helped my final rank, as pointed out by @jonaspalucibarbosa (https://www.kaggle.com/c/tabular-playground-series-jul-2021/discussion/256321). All of these extra variables were of course helpful prior to the leakage; post-leakage they didn't seem to help. 

Missing values: I estimated the missing (-200) values in the leaked data by using a two-stage estimation. @alexryzhkov post in the April competition was really helpful (https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/231738). In the second step I trained the model using a longer sample: the training data +  leaked data + the imputed missing values. It was also particularly helpful to include the targets as features, e.g. use benzene and nitrogen to predict carbon monoxide. 

Model: It was very helpful to use 5 gradient boosted tree models, each with a different seed. I averaged the predictions of the 5 models. I used the same models in each stage; I didn't have time to try using a different model for the second stage.

Finally, averaging my output with Alexander Ryzhkov's LightAutoML  (https://www.kaggle.com/alexryzhkov/tps-lightautoml-baseline-with-pseudolabels) improved my final rank from 2 to 1. 

Something I'm still thinking about: For the time period the training data was covering, there seemed to be a difference between the targets provided by Kaggle and the targets in the leaked data (a difference beyond the -200 values). The difference averaged zero over time and it appeared to be correlated with the sensors. I tried to improve my score by modeling this difference (I commented some of these lines in the code), but I didn't manage to improve it despite the correlation with the sensors. 

Thanks all again! I would appreciate any comments or questions!.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: