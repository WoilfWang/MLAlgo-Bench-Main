You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Restaurant_Revenue_Prediction_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
With over 1,200 quick service restaurants across the globe, TFI is the company behind some of the world's most well-known brands: Burger King, Sbarro, Popeyes, Usta Donerci, and Arby’s. They employ over 20,000 people in Europe and Asia and make significant daily investments in developing new restaurant sites.

Right now, deciding when and where to open new restaurants is largely a subjective process based on the personal judgement and experience of development teams. This subjective data is difficult to accurately extrapolate across geographies and cultures. 

New restaurant sites take large investments of time and capital to get up and running. When the wrong location for a restaurant brand is chosen, the site closes within 18 months and operating losses are incurred. 

Finding a mathematical model to increase the effectiveness of investments in new restaurant sites would allow TFI to invest more in other important business areas, like sustainability, innovation, and training for new employees. Using demographic, real estate, and commercial data, this competition challenges you to predict the annual restaurant sales of 100,000 regional locations.

TFI would love to hire an expert Kaggler like you to head up their growing data science team in Istanbul or Shanghai. You'd be tackling problems like the one featured in this competition on a global scale. See the job description here >>

##  Evaluation Metric:
Root Mean Squared Error (RMSE)
Submissions are scored on the root mean squared error. RMSE is very common and is a suitable general-purpose error metric. Compared to the Mean Absolute Error, RMSE punishes large errors:

$$\textrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2},$$

where y hat is the predicted value and y is the original value.

#### Submission File
For every restaurant in the dataset, submission files should contain two columns: Id and Prediction. 
The file should contain a header and have the following format:

    Id,Prediction
    0,1.0
    1,1.0
    2,1.0
    etc.

##  Dataset Description:
TFI has provided a dataset with 137 restaurants in the training set, and a test set of 100000 restaurants. The data columns include the open date, location, city type, and three categories of obfuscated data: Demographic data, Real estate data, and Commercial data. The revenue column indicates a (transformed) revenue of the restaurant in a given year and is the target of predictive analysis. 

#### File descriptions

    train.csv - the training set. Use this dataset for training your model. 
    test.csv - the test set. To deter manual "guess" predictions, Kaggle has supplemented the test set with additional "ignored" data. These are not counted in the scoring.
    sampleSubmission.csv - a sample submission file in the correct format

#### Data fields

    Id : Restaurant id. 
    Open Date : opening date for a restaurant
    City : City that the restaurant is in. Note that there are unicode in the names. 
    City Group: Type of the city. Big cities, or Other. 
    Type: Type of the restaurant. FC: Food Court, IL: Inline, DT: Drive Thru, MB: Mobile
    P1, P2 - P37: There are three categories of these obfuscated data. Demographic data are gathered from third party providers with GIS systems. These include population in any given area, age and gender distribution, development scales. Real estate data mainly relate to the m2 of the location, front facade of the location, car park availability. Commercial data mainly include the existence of points of interest including schools, banks, other QSR operators.
    Revenue: The revenue column indicates a (transformed) revenue of the restaurant in a given year and is the target of predictive analysis. Please note that the values are transformed so they don't mean real dollar values.

train.csv - column name: Id, Open Date, City, City Group, Type, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, revenue
test.csv - column name: Id, Open Date, City, City Group, Type, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37


## Dataset folder Location: 
../../kaggle-data/restaurant-revenue-prediction. In this folder, there are the following files you can use: train.csv, sampleSubmission.csv, test.csv

## Solution Description:

Here are the important steps of my solution:

1) Feature Engineering
i) Square root transformation was applied to the obfuscated P variables with maximum value >= 10, to make them into the same scale, as well as the target variable “revenue”.

ii) Random assignments of uncommon city levels to the common city levels in both training and test set, which I believe, diversified the geo location information contained in the city variable and in some of the obfuscated P variables.

Note: I discovered this to be helpful by chance. My intention was to assign uncommon city levels to their nearest common city levels. But it read the city levels differently on my laptop and on the server. It performed significantly better on the server. I am not 100% sure, but my explanation were given above.

iii) Missing value indicator for multiple P variables, i.e. P14 to P18, P24 to P27, and P30 to P37 was created to help differentiate synthetic and real test data.

Note: These variables were all zeroes on 88 out of 137 rows in the training set. The proportion was much less on the test set, i.e. those rows on which these variables were not zeroes at the same time has higher probability to be synthetic. 

iv) Type “MB”, which did not occur in training set, was changed to Type “DT” in test set.

v) Time / Age related information was also extracted, including open day, week, month and lasting years and days.

vi) Zeroes were treated as missing values and mice imputation was applied on training and test set separately.

2) Modelling and Selection Criteria
i ) Gradient boosting models were trained on the feature-engineered training set. I used R caret package and 10-fold cv repeated 10 times (default setting) to train the gbm models. The parameters grid used was simple to reduce over-fitting:

gbmGrid <- expand.grid(interaction.depth = c(6, 7, 8, 9),
n.trees = (3:7) * 10,
shrinkage = 0.05)

ii ) Two statistics were used to determine the model(s) to choose: training error and training error with outliers removed. The error limits are 3.7 * 10^12 and 1.4 * 10^12, respectively.

Note: I tested this strategy post-deadline, it was very effective choosing the "right" models. Around one in 15-20 models trained in step i) satisfied the two constraints. i.e. I trained about 200 models (using different seed) and 11 of them had training error and training error with outliers removed both lower than the limit I set. I randomly averaged 4 of them to make it more robust as a final model. These final models scored ~1718 to ~1735 privately and ~1675 to ~1707 publicly. (I guess taking average was more effective on the public data.) 

3) Possible Improvements
I read from the forum that dealing with outliers properly could improve scores, although I did not try it out myself. And in this situation, my strategy in 2) ii) might need modification.

4) Conclusion
I got lucky on this competition, while my true intention was to stabilize my performance on top 5%. As you guys can see, I am trying hard to stay in top 5% on two other competitions near its end. The techniques or methods I used here might not be a good strategy for other problems/competitions, and vice versa. I am learning a lot from the Kaggle forum and by participating in Kaggle competitions and gathering experience throughout these practices. Cheers!

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: