You are given a detailed description of a data science competition task, including the evaluation metric and a detailed description of the dataset. You are required to complete this competition using Python. 
Additionally, a general solution is provided for your reference, and you should implement this solution according to the given approach. 
You may use any libraries that might be helpful.
Finally, you need to generate a submission.csv file as specified.

## Task Description
Thank you to everyone who participated in and contributed to Season 3 Playground Series so far! 
With the same goal to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science, we will continue launching the Tabular Tuesday in July every Tuesday 00:00 UTC, with each competition running for 3 weeks. Again, these will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc. 
This Episode is a similar to the Kaggle/Zindi Hackathon that was held at the Kaggle@ICLR 2023: ML Solutions in Africa workshop in Rwanda, and builds on an ongoing partnership between Kaggle and Zindi to build community-driven impact across Africa. Zindi is a professional network for data scientists to learn, grow their careers, and get jobs. If you haven't done so recently, stop by Zindi and see what they're up to!

#### Predicting CO2 Emissions
The ability to accurately monitor carbon emissions is a critical step in the fight against climate change. Precise carbon readings allow researchers and governments to understand the sources and patterns of carbon mass output. While Europe and North America have extensive systems in place to monitor carbon emissions on the ground, there are few available in Africa. 

The objective of this challenge is to create a machine learning models using open-source CO2 emissions data from Sentinel-5P satellite observations to predict future carbon emissions.

These solutions may help enable governments, and other actors to estimate carbon emission levels across Africa, even in places where on-the-ground monitoring is not possible.


##  Evaluation Metric:
Root Mean Squared Error (RMSE)
Submissions are scored on the root mean squared error. RMSE is defined as:
$$
\textrm{RMSE} =  \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 }
$$
where \( \hat{y}_i \) is the predicted value and \( y_i \) is the original value for each instance \(i\).
Submission File
For each ID_LAT_LON_YEAR_WEEK row in the test set, you must predict the value for the target emission. The file should contain a header and have the following format:

    ID_LAT_LON_YEAR_WEEK,emission
    ID_-0.510_29.290_2022_00,81.94
    ID_-0.510_29.290_2022_01,81.94
    ID_-0.510_29.290_2022_02,81.94
    etc.


##  Dataset Description:
The objective of this challenge is to create machine learning models that use open-source emissions data (from Sentinel-5P satellite observations) to predict carbon emissions.
Approximately 497 unique locations were selected from multiple areas in Rwanda, with a distribution around farm lands, cities and power plants. The data for this competition is split by time; the years 2019 - 2021 are included in the training data, and your task is to predict the CO2 emissions data for 2022 through November.
Seven main features were extracted weekly from Sentinel-5P from January 2019 to November 2022. Each feature (Sulphur Dioxide, Carbon Monoxide, etc) contain sub features such as column_number_density which is the vertical column density at ground level, calculated using the DOAS technique. You can read more about each feature in the below links, including how they are measured and variable definitions. You are given the values of these features in the test set and your goal to predict CO2 emissions using time information as well as these features.

    Sulphur Dioxide - COPERNICUS/S5P/NRTI/L3_SO2
    Carbon Monoxide - COPERNICUS/S5P/NRTI/L3_CO
    Nitrogen Dioxide - COPERNICUS/S5P/NRTI/L3_NO2
    Formaldehyde - COPERNICUS/S5P/NRTI/L3_HCHO
    UV Aerosol Index - COPERNICUS/S5P/NRTI/L3_AER_AI
    Ozone - COPERNICUS/S5P/NRTI/L3_O3
    Cloud - COPERNICUS/S5P/OFFL/L3_CLOUD

Important: Please only use the data provided for this challenge as part of your modeling effort. Do not use any external data, including any data from Sentinel-5P not provided on this page.

Files

    train.csv - the training set
    test.csv - the test set; your task is to predict the emission target for each week at each location
    sample_submission.csv - a sample submission file in the correct format

## Dataset folder Location: 
../../kaggle-data/playground-series-s3e20.

## Solution Description:
I started this competition by building a pipeline to make CV and test different approaches. This pipeline consequently used each available year (2019, 2020, and 2021) for validation while the others were used for training. Finally, all the data was used for training and prediction of the 2022 emissions.

The core of my solution is using PCA. I decided to use 6 components.

Each of the 6 components is processed independently by some (abstract) algorithm, and the results are taken to inverse PCA. Finally, we have the predictions.

Each algorithm I used has a base estimator as a parameter, which is used to make a prediction. With such an architecture, it's possible to switch between them and optimize CV score.

✓ PCA1 Algorithm
This algorithm is very simple: it takes emission data only and processes it with an estimator. It was reported in many discussions that all other columns are not needed. So this is an implementation of this idea.

✓ PCA2 Algorithm
This algorithm is different: it uses all columns. PCA is applied separately to each of them, i.e., we have n_columns * n_pca_components input features.

✓ PCA_SARIMA Algorithm
Instead of simple estimators, I used 6 independent SARIMA models in the PCA1 Algorithm architecture.

The best algorithm (both local CV and private LB) was the PCA2 Algorithm.

I used Ridge/Lasso regression, RF, XGB, CatBoost, etc. There was not much difference between tree-based estimators, and they gave better results than linear regression. So I choose XGBRegressor(n_estimators = 300, max_depth = 4, learning_rate = 0.01, subsample = 0.5).




Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: