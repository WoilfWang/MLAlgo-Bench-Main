You are given a detailed description of a data science competition task, including the evaluation metric and a detailed description of the dataset. You are required to complete this competition using Python. 
Additionally, a general solution is provided for your reference, and you should implement this solution according to the given approach. 
You may use any libraries that might be helpful.
Finally, you need to generate a submission.csv file as specified.

## Task Description
As a global specialist in personal insurance, BNP Paribas Cardif serves 90 million clients in 36 countries across Europe, Asia and Latin America.
In a world shaped by the emergence of new uses and lifestyles, everything is going faster and faster. When facing unexpected events, customers expect their insurer to support them as soon as possible. However, claims management may require different levels of check before a claim can be approved and a payment can be made. With the new practices and behaviors generated by the digital economy, this process needs adaptation thanks to data science to meet the new needs and expectations of customers.

In this challenge, BNP Paribas Cardif is providing an anonymized database with two categories of claims:

claims for which approval could be accelerated leading to faster payments
claims for which additional information is required before approval

Kagglers are challenged to predict the category of a claim based on features available early in the process, helping BNP Paribas Cardif accelerate its claims process and therefore provide a better service to its customers.

##  Evaluation Metric:
The evaluation metric for this competition is Log Loss

$$log loss = -\frac{1}{N}\sum_{i=1}^N {(y_i\log(p_i) + (1 - y_i)\log(1 - p_i))}$$

where N is the number of observations, \\(log\\) is the natural logarithm, \\(y_{i}\\) is the binary target, and \\(p_{i}\\) is the predicted probability that \\(y_i\\) equals 1.
Note: the actual submitted predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

**Submission File**
For every observation in the test dataset, submission files should contain two columns: ID and PredictedProb. The file should contain a header and have the following format:
    
    ID,PredictedProb
    0,0.5
    1,0.5
    2,0.5
    7,0.5
    etc.

##  Dataset Description:
You are provided with an anonymized dataset containing both categorical and numeric variables available when the claims were received by BNP Paribas Cardif. All string type variables are categorical. There are no ordinal variables.
The "target" column in the train set is the variable to predict. It is equal to 1 for claims suitable for an accelerated approval.
The task is to predict a probability ("PredictedProb") for each claim in the test set.

**File descriptions**

    train.csv - the training set including the target
    test.csv - the test set without the target
    sample_submission.csv - a sample submission file in the correct format

## Dataset folder Location: 
../../kaggle-data/bnp-paribas-cardif-claims-management.

## Solution Description:
The following is the solution of Dexter's Lab, who won the competition with the score of 0.42037

We figured, that data unanonymizing was important and in the end we were able to figure out some of variables' meaning:

    v40 - date of observation
    v40-v50 - contract start date
    v50 - days since contract startdate (in other words, claimdate)
    v10 - contract term in months
    
and few others for remaining days of the contract:

    v12=v10*(365.25)/12-v50 
    v34=v10*(365.25)/12-v40

Now add this knowledge to the assumption that v22 is a customer and v56/v113 is a product type, 
and you will see very obvious patterns, that one contract may have several claims during contract lifespan (R: group_by(v22,v40-v50) %>% arrange(v50)):

    [v22,v40-v50]   [v50] [target]
    [ZLS 12840]   197      1
    [ZLS 12840]   962      1
    [ZLS 12840]  1437      1
    [ZLS 12840]  1498     NA
    [ZLS 12840]  1501      1
    [ZLS 12840]  1726      1
    [ZLS 12840]  1788     NA
    [ZLS 12840]  1882     NA
    [ZLS 12840]  2418      1
    [ZLS 12840]  3352     NA
    [ZLS 12840]  3370     NA
    [ZLS 12840]  3909     NA

So we could drop the categorical {v22} level i.i.d. assumption, and move to panel data structured categorical levels such as:

    {v22,v40-v50} => sort(v50), {v22,v40-v50,v56} => sort(v50), {v22} => sort(v40), etc; 
    
Just looking at the data this way, it was obvious that target value is very persistent for each {v22,v40-v50} level - 
i.e. if a time series starts with target=1, it usually ends with target=1; if it starts with target=0 it often ends with target=0 too; claims which target shifts 0->1 are quite rare (1->0 only few cases);

So in the end it all resulted to correctly imputing target sequence for each level - which we done using lag(target) and lead(target) variables. To our surprise, these variables were not overfitting LB, and in the end we made so many lag/lead variables that it was possible to drop v22 column, and didn't even use v22 mean target techniques disccussed in the forums, which probably many top teams did anyway.

Our best single xgboost model achieved 0.42347 public LB (0.42193 private), and the model takes itself only about half an hour to train on 12-thread cpu. To seal the deal, in last 2 weeks we experimented with other techniques and build few stack models with tens of different methodology models, such as nnets, linear SVM's, elastic nets, xgboosts with count:poisson, rank:pairwise, etc.

I personally enjoyed working with regularized greedy forests, which were almost on par with xgboost.
The role of ensembling may not as important as other competition though we have tried several diverse models. 

    - models with tsne feature from continous variables
    - models by levels of certain varaible (for example,var5) 
    - knn models on likehihood encoding varibles

**To sum up**: 

As most of you, we were stuck at 0.45x for a long time and it took 3-4 weeks of dedicated time of looking and exploring the data in Excel to end up with panel time-series data, which was the key to success.

Having such knowledge about the data could have gotten you to top10 without too much effort.
And for guys who wants to succeed - 

    a) when starting a new competition, create simple xgboost model and use feature importance to get a nice start to discovering important features - then stop making models and work with the data.
    b) do not underestimate the power of knowing what data you are working with
    c) dedicate some time for data exploration and try to understand how people visualize the data in the forums
    d) look for data patterns, especially if it has many categorical variables
    e) spend some time reading forums of past competitions, especially winning materials
    f) keep eyes on overfitting

"I found that the out-of-fold CV predictions of categorized lag features were very important. As far as I saw in the forum, many of the participants may have not created these features." This shook our beliefs and assumptions about data being i.i.d in {v22} levels, and it took us only 1 day to utilize this and claim the top1 rank for the rest of the competition:)


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: