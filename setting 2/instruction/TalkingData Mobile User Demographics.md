You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named TalkingData_Mobile_User_Demographics_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Nothing is more comforting than being greeted by your favorite drink just as you walk through the door of the corner café. While a thoughtful barista knows you take a macchiato every Wednesday morning at 8:15, it’s much more difficult in a digital space for your preferred brands to personalize your experience.

TalkingData, China’s largest third-party mobile data platform, understands that everyday choices and behaviors paint a picture of who we are and what we value. Currently, TalkingData is seeking to leverage behavioral data from more than 70% of the 500 million mobile devices active daily in China to help its clients better understand and interact with their audiences.

In this competition, Kagglers are challenged to build a model predicting users’ demographic characteristics based on their app usage, geolocation, and mobile device properties. Doing so will help millions of developers and brand advertisers around the world pursue data-driven marketing efforts which are relevant to their users and catered to their preferences.


##  Evaluation Metric:
Submissions are evaluated using the multi-class logarithmic loss. Each device has been labeled with one true class. For each device, you must submit a set of predicted probabilities (one for each class). The formula is then,
$$log loss = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}\log(p_{ij}),$$
where N is the number of devices in the test set, M is the number of class labels,  \\(log\\) is the natural logarithm, \\(y_{ij}\\) is 1 if device \\(i\\) belongs to class \\(j\\) and 0 otherwise, and \\(p_{ij}\\) is the predicted probability that observation \\(i\\) belongs to class \\(j\\).

The submitted probabilities for a given device are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum), but they need to be in the range of [0, 1]. In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).
Submission File
You must submit a csv file with the device id, and a probability for each class.

The 12 classes to predict are:

'F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+',  'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+'
The order of the rows does not matter. The file must have a header and should look like the following:

    device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+
    1234,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833
    5678,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833
    ...

##  Dataset Description:
In this competition, you are going to predict the demographics of a user (gender and age) based on their app download and usage behaviors. 

The Data is collected from TalkingData SDK integrated within mobile apps TalkingData serves under the service term between TalkingData and mobile app developers. Full recognition and consent from individual user of those apps have been obtained, and appropriate anonymization have been performed to protect privacy. Due to confidentiality, we won't provide details on how the gender and age data was obtained. Please treat them as accurate ground truth for prediction. 
The data schema can be represented in the following chart:

File descriptions

gender_age_train.csv, gender_age_test.csv - the training and test set

    group: this is the target variable you are going to predict

events.csv, app_events.csv - when a user uses TalkingData SDK, the event gets logged in this data. Each event has an event id, location (lat/long), and the event corresponds to a list of apps in app_events.

    timestamp: when the user is using an app with TalkingData SDK

app_labels.csv - apps and their labels, the label_id's can be used to join with label_categories

label_categories.csv - apps' labels and their categories in text

phone_brand_device_model.csv - device ids, brand, and models

phone_brand: note that the brands are in Chinese (translation courtesy of user fromandto) 

    三星 samsung
    天语 Ktouch
    海信 hisense
    联想 lenovo
    欧比 obi
    爱派尔 ipair
    努比亚 nubia
    优米 youmi
    朵唯 dowe
    黑米 heymi
    锤子 hammer
    酷比魔方 koobee
    美图 meitu
    尼比鲁 nibilu
    一加 oneplus
    优购 yougo
    诺基亚 nokia
    糖葫芦 candy
    中国移动 ccmc
    语信 yuxin
    基伍 kiwu
    青橙 greeno
    华硕 asus
    夏新 panosonic
    维图 weitu
    艾优尼 aiyouni
    摩托罗拉 moto
    乡米 xiangmi
    米奇 micky
    大可乐 bigcola
    沃普丰 wpf
    神舟 hasse
    摩乐 mole
    飞秒 fs
    米歌 mige
    富可视 fks
    德赛 desci
    梦米 mengmi
    乐视 lshi
    小杨树 smallt
    纽曼 newman
    邦华 banghua
    E派 epai
    易派 epai
    普耐尔 pner
    欧新 ouxin
    西米 ximi
    海尔 haier
    波导 bodao
    糯米 nuomi
    唯米 weimi
    酷珀 kupo
    谷歌 google
    昂达 ada
    聆韵 lingyun

sample_submission.csv - a sample submission file in the correct format

app_labels.csv - column name: app_id, label_id
phone_brand_device_model.csv - column name: device_id, phone_brand, device_model
events.csv - column name: event_id, device_id, timestamp, longitude, latitude
gender_age_train.csv - column name: device_id, gender, age, group
app_events.csv - column name: event_id, app_id, is_installed, is_active
gender_age_test.csv - column name: device_id
label_categories.csv - column name: label_id, category


## Dataset folder Location: 
../../kaggle-data/talkingdata-mobile-user-demographics. In this folder, there are the following files you can use: app_labels.csv, phone_brand_device_model.csv, events.csv, sample_submission.csv, gender_age_train.csv, app_events.csv, gender_age_test.csv, label_categories.csv

## Solution Description:
Hello, here I would like to share what we were doing all those weeks, and specially the last and leaky end of the last week. I'll give my view point and Danijel may add his viewpoint as well.
As usual in any competition, I started doing some exploratory analysis and my initial assumptions were that the ratio of usage of different apps would be predictive for age and gender. For example if the user has a lot of events associated to PokemonGo, war craft and other games, it's very likely that he is one of my coworkers! I even built this Kaggle script where I try to analyze ratio of apps type usage: https://www.kaggle.com/chechir/talkingdata-mobile-user-demographics/only-another-exploratory-analysis 
At the same time I saw the great script of Yibo: (https://www.kaggle.com/yibochen/talkingdata-mobile-user-demographics/xgboost-in-r-2-27217) and I copied his way to encode everything to 1/0 (event the ratios). Then I started to use a bunch of xgb models as well as a glmnet model, blending them all using multivariate regression (nnet package in R). I was doing reasonable well (around 20-30 place on the LB) when I saw the dune_dweller script (don't need to add the link!). At that time I was trying to learn Keras, so I used her feature engineering and plugged a keras model.. It had a great performance and boosted my score around the 17th position! For some reason I decided to share it on the Kaggle scripts: https://www.kaggle.com/chechir/talkingdata-mobile-user-demographics/keras-on-labels-and-brands. And our best model single model for devices with events is just that model with some new features and a more layers and regularization. It scored 2.23452 on the LB. 
The additional features to this model were:

TF-IDF of brand and model (for devices without events) 
TF-IDF of brand, model and labels (for devices with events)  
Frequency of brands and model names (that one produced a small but clear
improvement)  

And the parameters were: 
model = Sequential()
model.add(Dropout(0.4, input_shape=(num_columns,)))
model.add(Dense(n))
model.add(PReLU())
model.add(Dropout(0.30))
model.add(Dense(50, init='normal', activation='tanh'))
model.add(PReLU())
model.add(Dropout(0.20))
model.add(Dense(12, init='normal', activation='softmax'))
content_copy
Using the average test predictions for each fold model helped a lot here (I was using 10 folds)
Then we merged our teams with Danijel who was doing a very creative xgboost combination besides other things that he would explain much better than me. Together we started retraining some of our models on CV10 and then switched back to CV5 because of processing time reasons. For the ensemble weights we used the optim package in R (iterated 3 times) and also built differen ensembles for devices with events and devices without events (that was another great idea from Danijel). 

When the leak issue raised we were around 11th to 13th on the LB and we started to look where it was. My team mate Danijel was the man how built a clever matching script that combined with our best submission allowed us to fight for the top places in those crazy 3 last days. We found also that devices with events weren't taking advantage from the leak, so we only used leak models on non-events devices. 
To me this competition was a great experience, I learned a lot from my team mate Danijel,  and also from Dune_Dweller, Yibo and from fakeplastictrees and the dam leak!.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: