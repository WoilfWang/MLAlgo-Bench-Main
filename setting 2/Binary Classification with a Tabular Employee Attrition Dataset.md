You are given a detailed description of a data science competition task, including the evaluation metric and a detailed description of the dataset. You are required to complete this competition using Python. 
Additionally, a general solution is provided for your reference, and you should implement this solution according to the given approach. 

You may use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Binary_Classification_with_a_Tabular_Employee_Attrition_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the 2023 edition of Kaggle's Playground Series!

Thank you to everyone who participated in and contributed to last year's Tabular Playground Series. And many thanks to all those who took the time to provide constructive feedback! We're thrilled that there continues to be interest in these types of challenges, and we're continuing the series this year but with a few changes.

First, the series is getting upgraded branding. We've dropped "Tabular" from the name because, while we anticipate this series will still have plenty of tabular competitions, we'll also be having some other formats as well. You'll also notice freshly-upgraded (better looking and more fun!) banner and thumbnail images. 

Second, rather than naming the challenges by month and year, we're moving to a Season-Edition format. This year is Season 3, and each challenge will be a new Edition. We're doing this to have more flexibility. Competitions going forward won't necessarily align with each month like they did in previous years (although some might!), we'll have competitions with different time durations, and we may have multiple competitions running at the same time on occasion.

Regardless of these changes, the goals of the Playground Series remain the same—to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. We hope we continue to meet this objective!

To start the year with some fun, January will be the month of Tabular Tuesday. We're launching four week-long tabular competitions, with each starting Tuesday 00:00 UTC. These will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

**Synthetically-Generated Datasets**
Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

**Submission File**

For each EmployeeNumber in the test set, you must predict the probability for the target variable Attrition. The file should contain a header and have the following format:

    EmployeeNumber,Attrition
    1677,0.78
    1678,0.34
    1679,0.55
    etc.
    content_copy

##  Dataset Description:

The dataset for this competition (both train and test) was generated from a deep learning model trained on a Employee Attrition. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

**Files**

    train.csv - the training dataset; Attrition is the binary target
    test.csv - the test dataset; your objective is to predict the probability of positive Attrition
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, Attrition
test.csv - column name: id, Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager


## Dataset folder Location: 
../../kaggle-data/playground-series-s3e3. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
The following is the solution of Bill Cruise, ranked at the 1st position in this competition.

I hadn't even selected any submissions because I wasn't expecting to do well as I watched my public LB score slip. 😄
Many thanks to Khawaja Abaid, whose notebook Starting Strong - XGBoost + LightGBM + CatBoost was the basis of my own. Please go upvote Khawaja's notebook if you haven't already. My only big change was to add some feature engineering before training the same models. I had discussed it previously in Adding Risk Factors, but here's the final FE code from the winning version:

    df['MonthlyIncome/Age'] = df['MonthlyIncome'] / df['Age']
    
    df["Age_risk"] = (df["Age"] < 34).astype(int)
    df["HourlyRate_risk"] = (df["HourlyRate"] < 60).astype(int)
    df["Distance_risk"] = (df["DistanceFromHome"] >= 20).astype(int)
    df["YearsAtCo_risk"] = (df["YearsAtCompany"] < 4).astype(int)
    
    df['NumCompaniesWorked'] = df['NumCompaniesWorked'].replace(0, 1)
    df['AverageTenure'] = df["TotalWorkingYears"] / df["NumCompaniesWorked"]
    # df['YearsAboveAvgTenure'] = df['YearsAtCompany'] - df['AverageTenure']
    
    df['JobHopper'] = ((df["NumCompaniesWorked"] > 2) & (df["AverageTenure"] < 2.0)).astype(int)
    
    df["AttritionRisk"] = df["Age_risk"] + df["HourlyRate_risk"] + df["Distance_risk"] + df["YearsAtCo_risk"] + df['JobHopper']
    content_copy.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: