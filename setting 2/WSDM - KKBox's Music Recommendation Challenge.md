You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named WSDM_-_KKBox's_Music_Recommendation_Challenge_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
The 11th ACM International Conference on Web Search and Data Mining (WSDM 2018) is challenging you to build a better music recommendation system using a donated dataset from KKBOX. WSDM (pronounced "wisdom") is one of the the premier conferences on web inspired research involving search and data mining. They're committed to publishing original, high quality papers and presentations, with an emphasis on practical but principled novel models.

Not many years ago, it was inconceivable that the same person would listen to the Beatles, Vivaldi, and Lady Gaga on their morning commute. But, the glory days of Radio DJs have passed, and musical gatekeepers have been replaced with personalizing algorithms and unlimited streaming services.

While the public’s now listening to all kinds of music, algorithms still struggle in key areas. Without enough historical data, how would an algorithm know if listeners will like a new song or a new artist? And, how would it know what songs to recommend brand new users?

WSDM has challenged the Kaggle ML community to help solve these problems and build a better music recommendation system. The dataset is from KKBOX, Asia’s leading music streaming service, holding the world’s most comprehensive Asia-Pop music library with over 30 million tracks. They currently use a collaborative filtering based algorithm with matrix factorization and word embedding in their recommendation system but believe new techniques could lead to better results.
Winners will present their findings at the conference February 6-8, 2018 in Los Angeles, CA.  For more information on the conference, click here, and don't forget to check out the other KKBox/WSDM competition: KKBox Music Churn Prediction Challenge

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

Submission File

For each id in the test set, you must predict a probability for the target variable. The file should contain a header and have the following format:

    id,target
    2,0.3
    5,0.1
    6,1
    etc.

##  Dataset Description:
In this task, you will be asked to predict the chances of a user listening to a song repetitively after the first observable listening event within a time window was triggered. If there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, its target is marked 1, and 0 otherwise in the training set. The same rule applies to the testing set.

KKBOX provides a training data set consists of information of the first observable listening event for each unique user-song pair within a specific time duration. Metadata of each unique user and song pair is also provided. The use of public data to increase the level of accuracy of your prediction is encouraged.

The train and the test data are selected from users listening history in a given time period. Note that this time period is chosen to be before the WSDM-KKBox Churn Prediction time period. The train and test sets are split based on time, and the split of public/private are based on unique user/song pairs. 

#### Tables
train.csv

    msno: user id
    song_id: song id
    source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions.  For example, tab my library contains functions to manipulate the local storage, and tab search contains functions relating to search.
    source_screen_name: name of the layout a user sees. 
    source_type: an entry point a user first plays music on mobile apps. An entry point could be album, online-playlist, song .. etc. 
    target: this is the target variable. target=1 means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, target=0 otherwise . 

test.csv

    id: row id (will be used for submission)
    msno: user id
    song_id: song id
    source_system_tab: the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions.  For example, tab my library contains functions to manipulate the local storage, and tab search contains functions relating to search.
    source_screen_name: name of the layout a user sees. 
    source_type: an entry point a user first plays music on mobile apps. An entry point could be album, online-playlist, song .. etc. 

sample_submission.csv

sample submission file in the format that we expect you to submit

    id: same as id in test.csv
    target: this is the target variable. target=1 means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, target=0 otherwise . 

songs.csv

The songs. Note that data is in unicode. 

    song_id
    song_length: in ms
    genre_ids: genre category. Some songs have multiple genres and they are separated by |
    artist_name
    composer
    lyricist
    language

members.csv

user information. 

    msno
    city    
    bd: age. Note: this column has outlier values, please use your judgement. 
    gender    
    registered_via: registration method
    registration_init_time: format %Y%m%d
    expiration_date: format %Y%m%d

song_extra_info.csv

    song_id
    song name - the name of the song. 
    isrc - International Standard Recording Code, theoretically can be used as an identity of a song. However, what worth to note is, ISRCs generated from providers have not been officially verified; therefore the information in ISRC, such as country code and reference year, can be misleading/incorrect.  Multiple songs could share one ISRC since a single recording could be re-published several times.

train.csv - column name: msno, song_id, source_system_tab, source_screen_name, source_type, target
song_extra_info.csv - column name: song_id, name, isrc
test.csv - column name: id, msno, song_id, source_system_tab, source_screen_name, source_type
songs.csv - column name: song_id, song_length, genre_ids, artist_name, composer, lyricist, language
members.csv - column name: msno, city, bd, gender, registered_via, registration_init_time, expiration_date


## Dataset folder Location: 
../../kaggle-data/kkbox-music-recommendation-challenge. In this folder, there are the following files you can use: train.csv, song_extra_info.csv, test.csv, sample_submission.csv, songs.csv, members.csv

## Solution Description:
Thanks KKBox, Kaggle and WSDM for this competition!

The main hack of this competition is that we can generate features from future. Data is ordered chronologically. We can see, for example, listen or not the same artist in the future.

I use xgboost and catboost. Fit their on features among which was matrix_factorization, where I user LightFM. Catboost is better than xgboost for about 0.001.

I use last 35% of train for generation table of features, where the rest of the data was used as a history. For test – train is history. For validation analogous scheme, but the size of X_train, X_val was about 20% of data.

From hisotry it can be generated features using target. Other features can be generated using all data. I generate features in the context of categorical features (their pairs and triples):

1) mean of target from history by categorical features (pairs and triples)

2) count – the number of observations in all data by categorical features (pairs and triples)

3) regression – linear regression target by categorical features (msno, msno + genre_ids, msno+composer and so on)

4) time_from_prev_heard – time from last heard by categorical features (msno, msno + genre_ids, msno+composer and so on)

5) time_to_next_heard – time to next heard by categorical features (pairs and triples which include msno)

6) last_time_diff – time to the last heard by categorical features

7) part_of_unique_song – the share of unique song of artist that the user heard on all unique song of artist

8) matrix_factorization – LightFM. I use as a user msno in different context (source_type) and the msno was the feature for user.

I use last 35% for fit. Also I drop last 5% and again use 35%. Such a procedure could be repeated many times and this greatly improved the score (I did 6 times on validation and this gave an increase of 0.005, but in the end I did not have time to do everything and did only 2 times on test).

As a result, I fit xgboost and catboost on two parts of the data using and without the matrix_factorization feature. And finally I blend all 8 predictions.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: