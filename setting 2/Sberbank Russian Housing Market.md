You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Sberbank_Russian_Housing_Market_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Housing costs demand a significant investment from both consumers and developers. And when it comes to planning a budget—whether personal or corporate—the last thing anyone needs is uncertainty about one of their biggets expenses. Sberbank, Russia’s oldest and largest bank, helps their customers by making predictions about realty prices so renters, developers, and lenders are more confident when they sign a lease or purchase a building.

Although the housing market is relatively stable in Russia, the country’s volatile economy makes forecasting prices as a function of apartment characteristics a unique challenge. Complex interactions between housing features such as number of bedrooms and location are enough to make pricing predictions complicated. Adding an unstable economy to the mix means Sberbank and their customers need more than simple regression models in their arsenal.

In this competition, Sberbank is challenging Kagglers to develop algorithms which use a broad spectrum of features to predict realty prices. Competitors will rely on a rich dataset that includes housing data and macroeconomic patterns. An accurate forecasting model will allow Sberbank to provide more certainty to their customers in an uncertain economy.

##  Evaluation Metric:
Submissions are evaluated on the RMSLE between their predicted prices and the actual data. The target variable, called price_doc in the training set, is the sale price of each property.

Submission File

For each id in the test set, you must predict the price that the property sold for. The file should contain a header and have the following format:

    id,price_doc
    30474,7118500.44
    30475,7118500.44
    30476,7118500.44
    etc.

##  Dataset Description:
The aim of this competition is to predict the sale price of each property. The target variable is called price_doc in train.csv.

The training data is from August 2011 to June 2015, and the test set is from July 2015 to May 2016. The dataset also includes information about overall conditions in Russia's economy and finance sector, so you can focus on generating accurate price forecasts for individual properties, without needing to second-guess what the business cycle will do.

Data Files

    train.csv, test.csv: information about individual transactions. The rows are indexed by the "id" field, which refers to individual transactions (particular properties might appear more than once, in separate transactions). These files also include supplementary information about the local area of each property.
    macro.csv: data on Russia's macroeconomy and financial sector (could be joined to the train and test sets on the "timestamp" column)
    sample_submission.csv: an example submission file in the correct format
    data_dictionary.txt: explanations of the fields available in the other data files

As for train.csv and test.csv:

    price_doc: sale price (this is the target variable)
    id: transaction id
    timestamp: date of transaction
    full_sq: total area in square meters, including loggias, balconies and other non-residential areas
    life_sq: living area in square meters, excluding loggias, balconies and other non-residential areas
    floor: for apartments, floor of the building
    max_floor: number of floors in the building
    material: wall material
    build_year: year built
    num_room: number of living rooms
    kitch_sq: kitchen area
    state: apartment condition
    product_type: owner-occupier purchase or investment
    sub_area: name of the district

    The dataset also includes a collection of features about each property's surrounding neighbourhood, and some features that are constant across each sub area (known as a Raion). Most of the feature names are self explanatory

As for macro.csv, this file contains the following columns: timestamp and many other features.


## Dataset folder Location: 
../../kaggle-data/sberbank-russian-housing-market. In this folder, there are the following files you can use: train.csv, macro.csv, test.csv, sample_submission.csv

## Solution Description:
Our best submission was ensemble from several models from our individual solutions, which we'll describe in details below. 

We merged different subsets of predictions separately - e.g. each product type (Investment and OwnerOccupier) was predicted by different models.

Already at the moment of forming our team (as it turns out now) submission of ensemble from our models had private LB score 0.30704, which was enough for 1st place.

Public score however was enought only for 3rd place, so we kept trying hard to find any possible opportunities for improvements.

Having two good models on hand helped in this a lot. By analysing residuals of our predictions we found and fixed weak places of our models and also mistakes in data - that all helped to reach 2nd place in public LB and allowed to wait for revealing of private LB with hope for 1st place. Which turned out to be true.
Next we'll describe core concepts and main ideas from our individual solutions and approaches.

alijs approach

I used LightGbm for all my models. 
In quite early stage of this competition I discovered several important insights:

    Investment and OwnerOccupier product types were too different and using separate models for each of them gave better results.
    My CV for Investment type model was amost unusable, but CV for OwnerOccupier product type gave much better correlation to public score (not perfect, but quite usable for selecting features). 
    I used 5-fold CV with random shuffling on OwnerOccupier product type, comparing results for different seeds.
    Data cleaning was very important. I used two levels of data cleaning - some models used data with slight cleaning, others with more cleaning applied.
    I removed full_sq feature, because it was considered too important by my models, but it contained many mistakes in data. 
    Instead of using full_sq as feature directly I added the following features:
    feature_X = full_sq / mean_full_sq_for_group_X, where X were different categorical features, like num_room, build_year, floor, sub_area, etc.
    As my CV didn't include any time based validation, I didn't know how good my solution deals with trend/seasonality component, so I used several probing submissions to check level of mean for public test set.
    That way I discovered, that my models predictions mean was different than mean of public test set (in similar way as popular public kernels) and "magic numbers" helped to correct the difference.
    But unlike public kernel I was using 2 models - so I also discovered, that magic numbers for Investment and OwnerOccupier product types actually were quite differnt - 1.05 and 0.9 correspondingly.
    As I had usable CV only for OwnerOccupier product type and those models were stronger, I made predictions for Investment products using my OwnerOccupier model and used those predictions as feature for Investment models.

All those insights helped me to get to the 1st place in the middle stage of competition. I stayed at the top for some week or more, tried a lot of other things, but didn't succeed in any significant further improvements and started to loose my position.
When I felt down to 4th place, I decided to look for a team member and invited Evgeny (which turned out to be really good decision).
His models strongest part turned out to be Investments (which was weakest part of my solution), so merging our models gave good boost to our score.

Evgeny approach

I also used lightGBM as it much faster and perform better than xgboost now.
My approach was very different from alijs and it helped us when we combined our models.
Some ideas for start:

    I didn't predict full apartment prices directly, but prices per 1 square meter.
    I guessed that product_type "OwnerOccupier" was direct sales from developers of new buildings, and "Investment" type was usual second-hand market and not all sales were real invesment. Those very different markets and I made separate models for them.
    There are many low bad prices at the data. As they related to avoid tax purposes of sellers and we hadn't information about sellers enough to detect them, I decided to drop them.
    Fake prices were not only prices with 1-2-3 millions. In some areas 2 or 3 million were real prices, in other even 10 million were much lower than market. I used my investment model to find them (I guess that most of non-investment prices were real) and drop data were prediction and target were much different (2 times or more). And repeated process one more after first iteration to clean more. Totally I dropped near 10% of train set as data with bad prices.

If you would keep them during training you could get two type of troubles:

    during local validation better or lower accuracy for low prices data could mask changes of accuracy of good data and you could miss something useful when you fine-tune model.
    the boosting technology at the "linear regression" mode based on mean target values of each split of data. If you used low prices, they appeared at different splits randomly, for example, when you got less than average share of low prices at specific split your prediction could be higher than average level. If you removed bad prices, you could exclude this random factor and get higher generalization and accuracy of the model. The practice confirmed that - my investment part had even better score on private compared to public.

Data cleaning

I fixed some errors in features with squares, build year and max floor, but nothing special.
Also I filled some features like build year and num_room based on other apartments with same "address".

Main part

I saw two different tasks at this competition:

    to determine macro influence to prices
    to determine specific locations, building and apartment conditions to prices

The first task I solved partially - I use my investment model as a way to find macro components of training periods which helped me to scale all periods to one level (it was not precise, but much better than without it). After I dropped bad prices and scaled the rest by periods I got ability to use cross validation (8 folds split by each half year). The CV perform good enough to tune model - SD was between .007 and .008. CV score was around .09 for non-investment model and around .13 for investment part. The difference with leaderbord's scores was just huge penalty by bad prices for all participants.

I didn't managed to solve forecast part of macro task and just used downscale coefficients (they were called "magic", but there were just macro amendments to get average level of test prices). As I scaled train prices up in early train periods (according macro) and drop bad prices, my average level of predictions were high and I should to use lower downscale coefficients than used at public scripts. I tried some different values of coefficients on leaderboard, found good average levels for investment and non-investment parts and after that scaled my predictions to those average level without additional probing of LB.

I tried to find specific coefficients for each test month by probing LB. For non-investment part it were better to reduce discount at two first months and to increase discount at January16. It got me .0005 improvement before we merged to team, but when we tried it at our final ensemble impact was much lower - less than .0001. For investment part similar changes made score worse.

I tried to build model to get trend similar to coefficients that I got from LB probing, and found that we had not enough macro data. Shifted and scaled oil prices together with USDRUB courses and cpi-ppi were very similar, but additional data should be used also to good generalization and precise.

For invest part I used all data for training, but predict and check CV score only to invest data. My validation approach helped me to select right features and create new ones. My models had less than 50 features.
I had one early version what performed better on public LB and another that was better on my CV, but little bit worse on public LB. We used my latter version as part of our alternative second final submission and it performed better on private LB (trust your CV).

For non-investment part I used 2-stage approach - first I used all data for train and predict non-invest part for train and test. After that I used another model based on non-invest data and prediction from 1st stage.
For non-invest part I used additional price scaling for train periods. Prices of the most of new building grew much faster than general level. I decided to calculate additional smoothed scale factors for each non-investment addresses with more than 30 apartments in train.

My non-invest part was weaker than investment part. I think my approach was not enough precise as we had not much data, but I'm sure it could be good for further development.

Basmannoe-Savelki & Kuncevo - hidden troubles

When we compared our predictions, we found few addresses with high differences.

One of them were apartments at Basmannoe area with kremlin_km==2.90274. Our mean predicts were ~150000 and ~200000 rub per sqm. As these apartments were so close to Kremlin they should have high prices. But probed LB showed true level around 90000 per sqm where our score improved by .002. It was little bit upset and we started to research. Geo coordinates from Chippy script pointed to park were no buildings at all. Then we try to find apartments with the same full_sq (test set had 2 digits precision). And found that apartments at Savelki and both Krukovo (near Savelki) absolutely the same. All apartments at Krukovo were result of fix of Tverskoe issue. As Anastasia wrote that for some properties addresses were approximate, we guessed that all 4 addresses are one in reality and changed address and other features to Savelki. When we built separate model for this place, we got .0023 on public LB, what was quite high to 22 apartments.
After that we checked most of non-investment addresses without previous history in the train.
Next was 10 apartments at Kuncevo (.0006). And few others with smaller improve.
Totally we got around .004 from such type of corrections.



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: