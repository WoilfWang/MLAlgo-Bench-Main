You are given a detailed description of a data science competition task, including the evaluation metric and a detailed description of the dataset. You are required to complete this competition using Python. 
Additionally, a general solution is provided for your reference, and you should implement this solution according to the given approach. 
You may use any libraries that might be helpful.
Finally, you need to generate a submission.csv file as specified.

## Task Description
Instead of waking to overlooked "Do not disturb" signs, Airbnb travelers find themselves rising with the birds in a whimsical treehouse, having their morning coffee on the deck of a houseboat, or cooking a shared regional breakfast with their hosts.

New users on Airbnb can book a place to stay in 34,000+ cities across 190+ countries. By accurately predicting where a new user will book their first travel experience, Airbnb can share more personalized content with their community, decrease the average time to first booking, and better forecast demand.

In this recruiting competition, Airbnb challenges you to predict in which country a new user will make his or her first booking. Kagglers who impress with their answer (and an explanation of how they got there) will be considered for an interview for the opportunity to join Airbnb's Data Science and Analytics team.

Wondering if you're a good fit? Check out this article on how Airbnb scaled data science to all sides of their organization, and visit their careers page for more on Airbnb's mission to create a world that inspires human connection.

##  Evaluation Metric:
The evaluation metric for this competition is NDCG (Normalized discounted cumulative gain) @k where k=5. NDCG is calculated as:

$$DCG_k=\sum_{i=1}^k\frac{2^{rel_i}-1}{\log_2{\left(i+1\right)}},$$
$$nDCG_k=\frac{DCG_k}{IDCG_k},$$

where \\(rel_i\\) is the relevance of the result at position \\(i\\).
\\(IDCG_k\\) is the maximum possible (ideal) \\(DCG\\) for a given set of queries. All NDCG calculations are relative values on the interval 0.0 to 1.0.

For each new user, you are to make a maximum of 5 predictions on the country of the first booking. The ground truth country is marked with relevance = 1, while the rest have relevance = 0.

For example, if for a particular user the destination is FR, then the predictions become:

[ FR ]  gives a $(NDCG=\frac{2^{1}-1}{log_{2}(1+1)}=1.0$)

[ US, FR ] gives a $(DCG=\frac{2^{0}-1}{log_{2}(1+1)}+\frac{2^{1}-1}{log_{2}(2+1)}=\frac{1}{1.58496}=0.6309$) 

**Submission File**
For every user in the dataset, submission files should contain two columns: id and country. The destination country predictions must be ordered such that the most probable destination country goes first.

The file should contain a header and have the following format:
    id,country000am9932b,NDF000am9932b,US000am9932b,IT01wi37r0hw,FRetc.

##  Dataset Description:
In this challenge, you are given a list of users along with their demographics, web session records, and some summary statistics. You are asked to predict which country a new user's first booking destination will be. All the users in this dataset are from the USA.

There are 12 possible outcomes of the destination country: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'. Please note that 'NDF' is different from 'other' because 'other' means there was a booking, but is to a country not included in the list, while 'NDF' means there wasn't a booking.

The training and test sets are split by dates. In the test set, you will predict all the new users with first activities after 7/1/2014 (note: this is updated on 12/5/15 when the competition restarted). In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010. 

**File descriptions**

**train_users.csv** - the training set of users

**test_users.csv** - the test set of users

    id: user id
    date_account_created: the date of account creation
    timestamp_first_active: timestamp of the first activity, note that it can be earlier than date_account_created or date_first_booking because a user can search before signing up
    date_first_booking: date of first booking
    gender
    age
    signup_method
    signup_flow: the page a user came to signup up from
    language: international language preference
    affiliate_channel: what kind of paid marketing
    affiliate_provider: where the marketing is e.g. google, craigslist, other
    first_affiliate_tracked: whats the first marketing the user interacted with before the signing up
    signup_app
    first_device_type
    first_browser
    country_destination: this is the target variable you are to predict

sessions.csv - web sessions log for users

    user_id: to be joined with the column 'id' in users table
    action
    action_type
    action_detail
    device_type
    secs_elapsed
    
    countries.csv - summary statistics of destination countries in this dataset and their locations
    age_gender_bkts.csv - summary statistics of users' age group, gender, country of destination

sample_submission.csv - correct format for submitting your predictions

## Dataset folder Location: 
../../kaggle-data/airbnb-recruiting-new-user-bookings.

## Solution Description:
This is the solution of Branden Murray at the top-5 position in the private Leaderboard. The solution got the score of 0.88658.
The final model was an ensemble of 8 XGBs, 4NN (nolearn Lasagne), 1 GLMNET, and 1 RandomForest (using 'ranger' package in R). I used my out-of-fold CV predictions to create a second layer XGB model.

**Feature Engineering**:

Users: I extracted all the normal things from the dates. Then I created dummy variables for all of the categorical variables. Nothing fancy here.

Sessions: Filled in blanks with "NULL" so it was a new factor level.

For each user:

    Number of actions taken
    Number of unique Actions, ActionTypes, ActionDetails, Devices
    Sum, mean, min,
    max, median, s.d., skewness, kurtosis of seconds_elapsed
    Entropy of Actions, ActionTypes, ActionDetails, Devices, 
    Secs_elapsed of the first, second, last, second-to-last, and third-to-last 
    Actions Ratios of unique Actions, ActionTypes, ActionDetails, Devices
    Ratios of entropies

Then I casted the sessions frame so each level of Action, ActionType, ActionDetail, and Device had its own column. I did this twice, once using the count and the second time using the sum of seconds_elapsed. Sometimes I would use the raw count and sec_elapsed values for the actions, actiontTypes, etc. and sometimes I would convert them to proportions. Additionally I sometimes computed similarity/distance matrices, and used nearest neighbors to diffuse the matrices. Information on this can be found here.

I said in another thread that I tried using n-grams to capture the sequence of actions and it didn't really help at all, but my single model that scored the highest on the private board used n-grams, so maybe it did help a little.

Then I created 12 "helper" columns, one for each destination. Basically what I did was find the average of each of my features for each country_destination, then found for each country_destination which features were most different based on standard deviation. For example, say the countries were [NDF, US, FR, IT] and their averages for "feature_1" are [1.04, 5.40, 0.50, 0.98], then "feature_1" would be one of the features I use in creating a helper column for the US. I find 30 of these columns for each country and sum them together and that becomes the "helper" column.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: