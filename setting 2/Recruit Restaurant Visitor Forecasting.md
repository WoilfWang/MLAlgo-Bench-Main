You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Recruit_Restaurant_Visitor_Forecasting_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Running a thriving local restaurant isn't always as charming as first impressions appear. There are often all sorts of unexpected troubles popping up that could hurt business.

One common predicament is that restaurants need to know how many customers to expect each day to effectively purchase ingredients and schedule staff members. This forecast isn't easy to make because many unpredictable factors affect restaurant attendance, like weather and local competition. It's even harder for newer restaurants with little historical data.

Recruit Holdings has unique access to key datasets that could make automated future customer prediction possible. Specifically, Recruit Holdings owns Hot Pepper Gourmet (a restaurant review service), AirREGI (a restaurant point of sales service), and Restaurant Board (reservation log management software).

In this competition, you're challenged to use reservation and visitation data to predict the total number of visitors to a restaurant for future dates. This information will help restaurants be much more efficient and allow them to focus on creating an enjoyable dining experience for their customers.

##  Evaluation Metric:
Submissions are evaluated on the root mean squared logarithmic error.
The RMSLE is calculated as
$$\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 },$$
where:
\\(n\\) is the total number of observations \\(p_i\\) is your prediction of visitors\\(a_i\\) is the actual number of visitors \\(\log(x)\\) is the natural logarithm of \\(x\\)

Submission File

For every store and date combination in the test set, submission files should contain two columns: id and visitors.  The id is formed by concatenating the air_store_id and visit_date with an underscore. The file should contain a header and have the following format:

    id,visitors
    air_00a91d42b08b08d9_2017-04-23,0  
    air_00a91d42b08b08d9_2017-04-24,0  
    air_00a91d42b08b08d9_2017-04-25,0  
    etc.


##  Dataset Description:
In this competition, you are provided a time-series forecasting problem centered around restaurant visitors. The data comes from two separate sites:

Hot Pepper Gourmet (hpg): similar to Yelp, here users can search restaurants and also make a reservation online
AirREGI / Restaurant Board (air): similar to Square, a reservation control and cash register system

You must use the reservations, visits, and other information from these sites to forecast future restaurant visitor totals on a given date. The training data covers the dates from 2016 until April 2017. The test set covers the last week of April and May of 2017. The test set is split based on time (the public fold coming first, the private fold following the public) and covers a chosen subset of the air restaurants. Note that the test set intentionally spans a holiday week in Japan called the "Golden Week."
There are days in the test set where the restaurant were closed and had no visitors. These are ignored in scoring. The training set omits days where the restaurants were closed.

File Descriptions

This is a relational dataset from two systems. Each file is prefaced with the source (either air_ or hpg_) to indicate its origin. Each restaurant has a unique air_store_id and hpg_store_id. Note that not all restaurants are covered by both systems, and that you have been provided data beyond the restaurants for which you must forecast. Latitudes and Longitudes are not exact to discourage de-identification of restaurants.

air_reserve.csv

This file contains reservations made in the air system. Note that the reserve_datetime indicates the time when the reservation was created, whereas the visit_datetime is the time in the future where the visit will occur.

    air_store_id - the restaurant's id in the air system
    visit_datetime - the time of the reservation
    reserve_datetime - the time the reservation was made
    reserve_visitors - the number of visitors for that reservation

hpg_reserve.csv

This file contains reservations made in the hpg system.

    hpg_store_id - the restaurant's id in the hpg system
    visit_datetime - the time of the reservation
    reserve_datetime - the time the reservation was made
    reserve_visitors - the number of visitors for that reservation

air_store_info.csv

This file contains information about select air restaurants. Column names and contents are self-explanatory.

    air_store_id
    air_genre_name
    air_area_name
    latitude
    longitude

Note: latitude and longitude are the latitude and longitude of the area to which the store belongs
hpg_store_info.csv
This file contains information about select hpg restaurants. Column names and contents are self-explanatory.

hpg_store_id
hpg_genre_name
hpg_area_name
latitude
longitude

Note: latitude and longitude are the latitude and longitude of the area to which the store belongs

store_id_relation.csv

This file allows you to join select restaurants that have both the air and hpg system.

    hpg_store_id
    air_store_id

air_visit_data.csv

This file contains historical visit data for the air restaurants.

    air_store_id
    visit_date - the date
    visitors - the number of visitors to the restaurant on the date

sample_submission.csv

This file shows a submission in the correct format, including the days for which you must forecast.

    id - the id is formed by concatenating the air_store_id and visit_date with an underscore
    visitors- the number of visitors forecasted for the store and date combination

date_info.csv

This file gives basic information about the calendar dates in the dataset.

    calendar_date
    day_of_week
    holiday_flg - is the day a holiday in Japan

date_info.csv - column name: calendar_date, day_of_week, holiday_flg
air_store_info.csv - column name: air_store_id, air_genre_name, air_area_name, latitude, longitude
air_visit_data.csv - column name: air_store_id, visit_date, visitors
air_reserve.csv - column name: air_store_id, visit_datetime, reserve_datetime, reserve_visitors
store_id_relation.csv - column name: air_store_id, hpg_store_id
hpg_reserve.csv - column name: hpg_store_id, visit_datetime, reserve_datetime, reserve_visitors
hpg_store_info.csv - column name: hpg_store_id, hpg_genre_name, hpg_area_name, latitude, longitude


## Dataset folder Location: 
../../kaggle-data/recruit-restaurant-visitor-forecasting. In this folder, there are the following files you can use: date_info.csv, air_store_info.csv, air_visit_data.csv, air_reserve.csv, store_id_relation.csv, sample_submission.csv, hpg_reserve.csv, hpg_store_info.csv

## Solution Description:
plantsgo approach summary:

#### Models
I have three same models with the same features:

    model_1(step=14): use 14 days for target and slip 30 times,so I have 14*30 samples.
    model_2(step=28): use 28 days for target and slip 15 times,so I have 28*15 samples.
    model_3(step=42): use 42 days for target and slip 10 times,so I have 42*10 samples.
The reason is I don't want my model only focus on the days before or the days last.

#### Features:
1. visit_info:(21,35,63,140,280,350,420) days before groupby：
air_store_id,weekday,holiday,air_area_name,air_genre_name like:
(air_store_id,weekday),(air_store_id,holiday),(air_area_name,air_genre_name ,holiday) and so on.
2. reserve info:(35,63,140) days before groupby：
air_store_id,weekday,holiday,air_area_name,air_genre_name

#### Ensembel:
Use (xgb,lgb,nn) 0.7lgb+0.2xgb+0.1*nn: only improved 0.0002 offline.

0.334model_1+0.333model_2+0.333*model_3:improved 0.002 offline.



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: