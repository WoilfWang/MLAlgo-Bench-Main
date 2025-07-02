You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Binary_Classification_with_a_Tabular_Reservation_Cancellation_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the 2023 edition of Kaggle's Playground Series!

Thank you to everyone who participated in and contributed to last year's Tabular Playground Series. And many thanks to all those who took the time to provide constructive feedback! We're thrilled that there continues to be interest in these types of challenges, and we're continuing the series this year but with a few changes.

First, the series is getting upgraded branding. We've dropped "Tabular" from the name because, while we anticipate this series will still have plenty of tabular competitions, we'll also be having some other formats as well. You'll also notice freshly-upgraded (better looking and more fun!) banner and thumbnail images. 

Second, rather than naming the challenges by month and year, we're moving to a Season-Edition format. This year is Season 3, and each challenge will be a new Edition. We're doing this to have more flexibility. Competitions going forward won't necessarily align with each month like they did in previous years (although some might!), we'll have competitions with different time durations, and we may have multiple competitions running at the same time on occasion.

Regardless of these changes, the goals of the Playground Series remain the same—to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. We hope we continue to meet this objective!

With the great start and participation in January, we will continue launching the Tabular Tuesday in February every Tuesday 00:00 UTC, with each competition running for 2 weeks instead. Again, these will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.
### Submission File
For each id in the test set, you must predict the value for the target booking_status. The file should contain a header and have the following format:

    id,booking_status
    42100,0
    42101,1
    42102,0
    etc.


##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Reservation Cancellation Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

### Files

    train.csv - the training dataset; booking_status is the target (e.g., whether the reservation was cancelled)
    test.csv - the test dataset; your objective is to predict booking_status
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan, required_car_parking_space, room_type_reserved, lead_time, arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests, booking_status
test.csv - column name: id, no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan, required_car_parking_space, room_type_reserved, lead_time, arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests


## Dataset folder Location: 
../../kaggle-data/playground-series-s3e7. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
### Feature Engineering
Although all of the data was numeric, a look at the data description revealed that the features type_of_meal_plan, room_type_reserved, and market_segment_type were actually categorical in nature. I handled these variables in 3 different ways:

    Leaving them as is
    One-hot encoding them
    Marking them as categorical features (used for LGBM)

I played around with creating additional features, but I didn't find anything that improved my CV significantly. 
### Data Leakage
Within the train and test set, there were 1531 pairs of records that were duplicates if you dropped booking status. 

    562 of those pairs were in the train set, and upon closer inspection, each of those pairs had opposite booking statuses. 
    253 of those pairs were in the test set. 
    The remaining 716 pairs included 1 record in the train set and 1 record in the test set. 

Out of curiosity, I tried modifying a submission by setting the predictions for those 716 records in the test set to the opposite of their corresponding record in the train set, and got a boost of +.014 to my leaderboard score. I naively hoped to keep this secret to myself, but it was quickly discovered by the rest of the community 😅 However, I do have to thank @siukeitin for their suggestion to predict .5 for the 253 pairs of duplicates in the test set. This boosted my score by +.003, and I don't believe this trick was as well-known.

Additionally, I removed from the train set the 562 duplicate pairs, plus the 716 records that had a duplicate in the test set, as I found that doing so improved my score on a holdout set I constructed to test this. I think this only resulted in a minor increase in my private score though. However, I do think that perhaps removing the train duplicate pairs helped make my CV more reliable.

### Data Cleaning
When running adversarial validation between the train and original datasets, I plotted for the original dataset the predicted probability of belonging to the train set and got a bimodal distribution like this:

This led me to experiment with removing the chunk of the original dataset that looked the least like the train set. I ended up using models that both included the entire original dataset and models that removed ~17% of the original dataset that looked the least like the train set.

I also played around with things like fixing dates, but didn't see significant improvement in CV score from doing so.

### Modeling
My CV setup was a stratified 3-fold with 3 repeats. For creating submissions, predictions from each fold were averaged. When tuning hyperparameters, one thing I noticed was that increased tree expressiveness seemed to improve performance to a much greater degree than what I had been used to. This meant using XGBoost with the 'exact' algorithm instead of 'hist' and max_depths around 12-13, and using LGBM with larger max_leaves and max_bin values.

My winning submission was an average of the following 4 XGB submissions and 2 LGBM submissions:

    XGB
    XGB with categorical variables one-hot-encoded
    XGB with reduced original dataset
    XGB with categorical variables one-hot-encoded and reduced original dataset
    LGBM with reduced original dataset
    LGBM with categorical features and reduced original dataset.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: