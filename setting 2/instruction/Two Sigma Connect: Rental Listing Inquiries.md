You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Two_Sigma_Connect:_Rental_Listing_Inquiries_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Finding the perfect place to call your new home should be more than browsing through endless listings. RentHop makes apartment search smarter by using data to sort rental listings by quality. But while looking for the perfect apartment is difficult enough, structuring and making sense of all available real estate data programmatically is even harder. Two Sigma and RentHop, a portfolio company of Two Sigma Ventures, invite Kagglers to unleash their creative engines to uncover business value in this unique recruiting competition.


Two Sigma invites you to apply your talents in this recruiting competition featuring rental listing data from RentHop. Kagglers will predict the number of inquiries a new listing receives based on the listing’s creation date and other features. Doing so will help RentHop better handle fraud control, identify potential listing quality issues, and allow owners and agents to better understand renters’ needs and preferences.

Two Sigma has been at the forefront of applying technology and data science to financial forecasts. While their pioneering advances in big data, AI, and machine learning in the financial world have been pushing the industry forward, as with all other scientific progress, they are driven to make continual progress. This challenge is an opportunity for competitors to gain a sneak peek into Two Sigma's data science work outside of finance.


##  Evaluation Metric:
Submissions are evaluated using the multi-class logarithmic loss. Each listing has one true class. For each listing, you must submit a set of predicted probabilities (one for every listing). The formula is then,
$$log loss = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}\log(p_{ij}),$$

where N is the number of listings in the test set, M is the number of class labels (3 classes),  \\(log\\) is the natural logarithm, \\(y_{ij}\\) is 1 if observation \\(i\\) belongs to class \\(j\\) and 0 otherwise, and \\(p_{ij}\\) is the predicted probability that observation \\(i\\) belongs to class \\(j\\).

The submitted probabilities for a given listing are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

Submission File

You must submit a csv file with the listing_id, and a probability for each class.
The order of the rows does not matter. The file must have a header and should look like the following:

    listing_id,high,medium,low
    7065104,0.07743170693194379,0.2300252644876046,0.6925430285804516
    7089035,0.0, 1.0, 0.0
    ...

##  Dataset Description:
In this competition, you will predict how popular an apartment rental listing is based on the listing content like text description, photos, number of bedrooms, price, etc. The data comes from renthop.com, an apartment listing website. These apartments are located in New York City.

The target variable, interest_level, is defined by the number of inquiries a listing has in the duration that the listing was live on the site. 

File descriptions

    train.json - the training set
    test.json - the test set
    sample_submission.csv - a sample submission file in the correct format
    images_sample.zip - listing images organized by listing_id (a sample of 100 listings)
    Kaggle-renthop.7z - (optional) listing images organized by listing_id. Total size: 78.5GB compressed. Distributed by BitTorrent (Kaggle-renthop.torrent). 

Data fields

    bathrooms: number of bathrooms
    bedrooms: number of bathrooms
    building_id
    created
    description
    display_address
    features: a list of features about this apartment
    latitude
    listing_id
    longitude
    manager_id
    photos: a list of photo links. You are welcome to download the pictures yourselves from renthop's site, but they are the same as imgs.zip. 
    price: in USD
    street_address
    interest_level: this is the target variable. It has 3 categories: 'high', 'medium', 'low'

## Dataset folder Location: 
../../kaggle-data/two-sigma-connect-rental-listing-inquiries. In this folder, there are the following files you can use: images_sample, test.json, sample_submission.csv, Kaggle-renthop.torrent, train.json

## Solution Description:
First of all, congrats to all top finishes again! And THANK YOU to everybody who were selfless with sharing their insights/kernels, I would do much worse without you! Also, Marios for sharing stacknet and leak feature to make kaggle a better place. :)

I started this competition hoping I could do something interesting with the images. But after playing with CNN for 2 weeks I gave up. :( But I did find the leak along the way, so not too bad. :)

My approach is fairly simple. People who have teamed up with me before would probably know that I prefer doing things in a "systematic" way (mainly because I am lazy……)

#### Feature Engineering Part I
Except for basic features, all my features were just "encoding other categorical features". That is, I used features (continuous & categorical) to encode some "important" features (especially, manager_id).

Let me show you what I mean by that:
```python
def get_stats(train_df, test_df, target_column, group_column = 'manager_id'):
'''
target_column: numeric columns to group with (e.g. price, bedrooms, bathrooms)
group_column: categorical columns to group on (e.g. manager_id, building_id)
'''
train_df['row_id'] = range(train_df.shape[0])
test_df['row_id'] = range(test_df.shape[0])
train_df['train'] = 1
test_df['train'] = 0
all_df = train_df[['row_id', 'train', target_column, group_column]].append(test_df[['row_id','train',
                                                                                    target_column, group_column]])
grouped = all_df[[target_column, group_column]].groupby(group_column)
the_size = pd.DataFrame(grouped.size()).reset_index()
the_size.columns = [group_column, '%s_size' % target_column]
the_mean = pd.DataFrame(grouped.mean()).reset_index()
the_mean.columns = [group_column, '%s_mean' % target_column]
the_std = pd.DataFrame(grouped.std()).reset_index().fillna(0)
the_std.columns = [group_column, '%s_std' % target_column]
the_median = pd.DataFrame(grouped.median()).reset_index()
the_median.columns = [group_column, '%s_median' % target_column]
the_stats = pd.merge(the_size, the_mean)
the_stats = pd.merge(the_stats, the_std)
the_stats = pd.merge(the_stats, the_median)

the_max = pd.DataFrame(grouped.max()).reset_index()
the_max.columns = [group_column, '%s_max' % target_column]
the_min = pd.DataFrame(grouped.min()).reset_index()
the_min.columns = [group_column, '%s_min' % target_column]

the_stats = pd.merge(the_stats, the_max)
the_stats = pd.merge(the_stats, the_min)

all_df = pd.merge(all_df, the_stats)

selected_train = all_df[all_df['train'] == 1]
selected_test = all_df[all_df['train'] == 0]
selected_train.sort_values('row_id', inplace=True)
selected_test.sort_values('row_id', inplace=True)
selected_train.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)
selected_test.drop([target_column, group_column, 'row_id', 'train'], axis=1, inplace=True)

return np.array(selected_train), np.array(selected_test)
```

This function just does the encoding with count, min, max, std, mean and median of the continuous features. So with this, I can do something like this:

```python
selected_manager_id_proj = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'bad_addr', 'listing_id',
                   'month', 'day', 'weekday', 'day_of_year', 'hour', 'num_features', 'num_desc',
                   'bed_to_bath', 'price_per_bed', 'price_per_bath', 'bldg_count', 'zero_bldg', 'total_room', 'room_diff',
                   'photo_count', 'latitude_grid', 'longitude_grid', 'lat_long_grid']

for target_col in selected_manager_id_proj:
    tmp_train, tmp_test = get_stats(train_df, test_df, target_column=target_col)
    train_stack_list.append(tmp_train)
    test_stack_list.append(tmp_test)

selected_bedrooms_proj = ['price', 'listing_id', 'month', 'day', 'weekday', 'day_of_year', 'hour', 'num_features', 'bldg_count', 'zero_bldg']

for target_col in selected_bedrooms_proj:
    tmp_train, tmp_test = get_stats(train_df, test_df, target_column=target_col, group_column='bedrooms')
    train_stack_list.append(tmp_train)
    test_stack_list.append(tmp_test)

```

And these two pieces are mostly what contributed to my best single model. I also did encoding with bathrooms, clusters based on lat/lon, etc. They also contributed a bit but not much.

I also tried encoding categorical with categorical, like encoding manager_id with building_id (one hot encoded/frequency/tfidf and then dimension reduction). They didn't contribute in the end (however might bring in some diversity for ensemble).

#### Feature Engineering Part II
Almost the same as part I but with leaky label encoding, similar to "manager_skill" mentioned in kernels. I did this because I found that leaky label encoding hurt my scores if I had rich representation of manager_id. But they alone had improvement and although cv was overfitted by ~0.0003, it was consistent. So I decided to generate another feature set starting with them. So basically I did something like this:

```python
for c in ['bathrooms', 'bedrooms','zero_bldg', 'latitude_grid', 'longitude_grid', 'lat_long_grid', 'manager_id', 'building_id']:
    tmp_train, tmp_test = get_label_encoder(c, train_df = train_df, test_df = test_df)
    train_fea_list.append(tmp_train)
    test_fea_list.append(tmp_test)
for target_col in ['price', 'num_features', 'listing_id', 'bedrooms', 'bathrooms']:
    for group_col in ["cluster_1", "cluster_2", "street_address", "manager_id"]:
        tmp_train, tmp_test = get_label_inter_stats(train_df, test_df, target_column=target_col, group_column=group_col)
        train_fea_list.append(tmp_train)
        test_fea_list.append(tmp_test)
```

And then I started adding features from part I group by group until no improvement could be seen.

In the end, I got 0.507 with xgboost on feature part I and 0.511 on feature part II on public LB. So overall my feature sets aren't that good.

#### First Layer Models
My stacking was just the typical 3 layer modeling. And unsurprisingly, for first layer, lightgbm, xgboost and nn are the main models. All trained with classification and regression. I found that tuning parameters didn't help much with my features so I didn't tune parameters much. My nn was about 0.03 worse than tree boosting models, and I couldn't improve it any further. Curious if that is the case for you, so please comment if you know better ways to tune NN. And random forest was very bad in my case. Like 0.58 ish.

And then I just added other models with stacknet, basically trained stacknet and then only took the first layer output. I didn't tune the parameters/models. All default setting from Marios' example script in github!! Many thanks!

I would expect if I included model one by one and fine tuned them a bit, I could see a bit more improvement. But fitting once and you have 10 models was just too tempting. :) If I had one more week to work on this, I would probably do it tho.

I did exactly the same for feature part I and feature part II

#### Second Layer Models
Again, lightgbm and nn (classification only). Including random forest had a tiny bit improvement.

#### Third Layer Models
weighted average of ensembles of models on feature set I & II.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: