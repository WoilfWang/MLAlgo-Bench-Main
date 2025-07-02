You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Prudential_Life_Insurance_Assessment_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Picture this. You are a data scientist in a start-up culture with the potential to have a very large impact on the business. Oh, and you are backed up by a company with 140 years' business experience.

Curious? Great! You are the kind of person we are looking for.

Prudential, one of the largest issuers of life insurance in the USA, is hiring passionate data scientists to join a newly-formed Data Science group solving complex challenges and identifying opportunities. The results have been impressive so far but we want more. 

## The Challenge
In a one-click shopping world with on-demand everything, the life insurance application process is antiquated. Customers provide extensive information to identify risk classification and eligibility, including scheduling medical exams, a process that takes an average of 30 days.

The result? People are turned off. That’s why only 40% of U.S. households own individual life insurance. Prudential wants to make it quicker and less labor intensive for new and existing customers to get a quote while maintaining privacy boundaries.

By developing a predictive model that accurately classifies risk using a more automated approach, you can greatly impact public perception of the industry.

The results will help Prudential better understand the predictive power of the data points in the existing assessment, enabling us to significantly streamline the process.

##  Evaluation Metric:
Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0.

The response variable has 8 possible ratings.  Each application is characterized by a tuple (ea,eb), which corresponds to its scores by Rater A (actual risk) and Rater B (predicted risk).  The quadratic weighted kappa is calculated as follows.

First, an N x N histogram matrix O is constructed, such that Oi,j corresponds to the number of applications that received a rating i by A and a rating j by B. An N-by-N matrix of weights, w, is calculated based on the difference between raters' scores:

$$w_{i,j} = \frac{\left(i-j\right)^2}{\left(N-1\right)^2}$$

An N-by-N histogram matrix of expected ratings, E, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that E and O have the same sum.

From these three matrices, the quadratic weighted kappa is calculated as: 
$$\kappa=1-\frac{\sum_{i,j}w_{i,j}O_{i,j}}{\sum_{i,j}w_{i,j}E_{i,j}}.$$

#### Submission File
For each Id in the test set, you must predict the Response variable. The file should contain a header and have the following format:

    Id,Response1,4
    3,8
    4,3
    etc.

##  Dataset Description:
In this dataset, you are provided over a hundred variables describing attributes of life insurance applicants. The task is to predict the "Response" variable for each Id in the test set. "Response" is an ordinal measure of risk that has 8 levels.
File descriptions

train.csv - the training set, contains the Response values
test.csv - the test set, you must predict the Response variable for all rows in this file
sample_submission.csv - a sample submission file in the correct format

Data fields



Variable
Description


Id
A unique identifier associated with an application.


Product_Info_1-7
A set of normalized variables relating to the product applied for


Ins_Age
Normalized age of applicant


Ht
Normalized height of applicant


Wt
Normalized weight of applicant


BMI
Normalized BMI of applicant


Employment_Info_1-6
A set of normalized variables relating to the employment history of the applicant.


InsuredInfo_1-6
A set of normalized variables providing information about the applicant.


Insurance_History_1-9
A set of normalized variables relating to the insurance history of the applicant.


Family_Hist_1-5
A set of normalized variables relating to the family history of the applicant.


Medical_History_1-41
A set of normalized variables relating to the medical history of the applicant.


Medical_Keyword_1-48
A set of dummy variables relating to the presence of/absence of a medical keyword being associated with the application.


Response
This is the target variable, an ordinal variable relating to the final decision associated with an application



The following variables are all categorical (nominal):

Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41

The following variables are continuous:

Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5

The following variables are discrete:

Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32
Medical_Keyword_1-48 are dummy variables.

## Dataset folder Location: 
../../kaggle-data/prudential-life-insurance-assessment. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Hello all, here's my writeup. Hope you find it insightful (I certainly did learn a lot in the course of the competition)!
Feature engineering:

create dummy vars for Product_Info_2 (keep everything else as
numeric)
calculate sum of all Medical_Keyword columns
for each binary keyword-value pair, calculate the mean of the target
variable, then for each observation take the mean and the minimum of
the keyword-value-meantargets

Modeling:

for i in 1 to 7: build an XGBClassifier to predict the probability
that the observation has a Response value higher than i (for each of
the seven iterations, the keyword-value-meantarget variables were
calculated for that target variable)
for each observation, take the sum of these seven predicted
probabilities as the overall prediction
this yields quite a bit better correlation with the target variable
(and thus good raw material for calibration) than using an XGB
regressor

Calibration:

the aim is to find the boundaries that maximize the kappa score
boundaries are initialized according to the original Response
distribution of the training dataset
then in a step, for all boundaries, possible boundary values are
examined in a small range around the current boundary value and the
boundary is set to the value which gives the most improvement in
kappa (independently of the other boundaries - this was surprising
that it worked so well)
steps are repeated until none of the boundaries are changed during a
step
it is a quite naive algorithm, but it turned out to be fairly robust
and efficient
this was done on predictions generated by repeated crossvalidation
using the XGBClassifier combo

Variable split calibration:

the difference here is that the crossvalidated preds are split into two
subsets, based on some binary variable value (eg. a Medical_Keyword
variable) of the observations
calibration then takes place for the two subsets separately (but with a kappa objective calculated over the entire set), in the manner described above
I didn't find an exact rule for picking a good splitting variable
(strong correlation with Response seems to be necessary, but does not
guarantee a good split), so I tried several (some of which were
better than non-splitting calibration, others were worse)
for example, some good ones were: Medical_History_23,
Medical_History_4, InsuredInfo6
also tried splitting into more than 2 subsets, without much success

Ensembling:

disregarding the combination of the 7 XGBClassifiers, the only
ensembling I did was creating some combined solutions by taking the
median predictions of a small number of other solutions

Evaluating calibrations:

K-fold crossvalidation, but with an important twist: each test fold
was "cross-validated" again to imitate public/private test set split
(the inner crossvalidation had a k of 3 to approximate the 30-70
leaderboard split)
this yielded a very interesting insight: given two calibrations with
roughly equal average performance (over all folds), if calibration A
does better on the public test set, calibration B is very likely to
outperform A on the private set (this appears to be a quirk of the
kappa metric)
accordingly, I picked the solutions which ranked #2 and #5 on the
public leaderboard, since these both had very strong average performance
in crossvalidation but slightly underperformed on the public
leaderboard

Final results:

as it turned out, despite having the right idea about public/private
error, I underestimated some solutions which had relatively weak
average performance in crossvalidation but ended up doing extremely
well on private
I did not select my best private submission for the final two
(highest private score was 0.68002)
out of my 11 'high-tech' (that is, using all the modeling and
calibration techniques listed above) submissions, 5 were good enough
for 1st place on the private board, 4 would place 2nd, one would
reach 6th, and the worst would yield 7th place (at least I can say
that I had no intention of picking any of the latter two)
if my calculations are right, randomly selecting two out of the 11
would have resulted in 1st place with a probability of ~72.7 %

Gábor.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: