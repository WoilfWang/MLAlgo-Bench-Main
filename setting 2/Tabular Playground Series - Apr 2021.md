You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_Apr_2021_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Your task is to predict whether or not a passenger survived the sinking of the Synthanic (a synthetic, much larger dataset based on the actual Titanic dataset). For each PasengerId row in the test set, you must predict a 0 or 1 value for the Survived target.

Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly. 

In order to have a more consistent offering of these competitions for our community, we're trying a new experiment in 2021. We'll be launching month-long tabular Playground competitions on the 1st of every month and continue the experiment as long as there's sufficient interest and participation.

The goal of these competitions is to provide a fun, and approachable for anyone, tabular dataset.  These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you. We encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals. 

The dataset is used for this competition is synthetic but based on a real dataset (in this case, the actual Titanic data!) and generated using a CTGAN. The statistical properties of this dataset are very similar to the original Titanic dataset, but there's no way to "cheat" by using public labels for predictions. How well does your model perform on truly private test labels?

##  Evaluation Metric:
Your score is the percentage of passengers you correctly predict. This is known as accuracy.

Submission File

You should submit a csv file with exactly 100,000 rows plus a header row. Your submission will show an error if you have extra columns or extra rows.

The file should have exactly 2 columns:

    PassengerId (sorted in any order)
    Survived (contains your binary predictions: 1 for survived, 0 for deceased)

You can download an example submission file (sample_submission.csv) on the Data page

    PassengerId,Survived
    100000,0
    100001,1
    100002,0
    etc.

##  Dataset Description:
Overview

The dataset is used for this competition is synthetic but based on a real dataset (in this case, the actual Titanic data!) and generated using a CTGAN. The statistical properties of this dataset are very similar to the original Titanic dataset, but there's no way to "cheat" by using public labels for predictions. How well does your model perform on truly unseen data?

The data has been split into two groups:

    training set (train.csv)
    test set (test.csv)

The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use  feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Synthanic.

Data Dictionary

    Variable Definition Key
    survival Survival 0 = No, 1 = Yes
    pclass Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
    sex Sex
    Age Age in years
    sibsp # of siblings / spouses aboard the Titanic
    parch # of parents / children aboard the Titanic
    ticket Ticket number
    fare Passenger fare
    cabin Cabin number
    embarked Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton


Variable Notes

    pclass: A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower 
    age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5 
    sibsp: The dataset defines family relations in this way... 
    Sibling = brother, sister, stepbrother, stepsister 
    Spouse = husband, wife (mistresses and fiancés were ignored) 
    parch: The dataset defines family relations in this way... 
    Parent = mother, father Child = daughter, son, stepdaughter, stepson 
    Some children travelled only with a nanny, therefore parch=0 for them.

train.csv - column name: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
test.csv - column name: PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked


## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-apr-2021. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
#### Data Processing
1. **Age Imputation**: Missing values in the 'Age' column are filled with the mean age of the dataset.

2. **Cabin Processing**: Missing values in the 'Cabin' column are replaced with 'X', and only the first letter of each cabin entry is retained.

3. **Ticket Processing**: Missing values in the 'Ticket' column are replaced with 'X'. The ticket entries are split by spaces, and only the first part is kept unless the ticket is missing, in which case 'X' is used.

4. **Fare Imputation and Transformation**: Missing values in the 'Fare' column are filled with the median fare for each passenger class ('Pclass'). The fare values are then transformed using the natural logarithm to reduce skewness.

5. **Embarked Imputation**: Missing values in the 'Embarked' column are filled with 'X'.

6. **Name Processing**: Only the surname is extracted from the 'Name' column.

7. **Encoding and Scaling**:
   - **Label Encoding**: The 'Name', 'Ticket', and 'Sex' columns are label encoded to convert categorical data into numerical format.
   - **One-Hot Encoding**: The 'Cabin' and 'Embarked' columns are one-hot encoded to create binary columns for each category.
   - **Standard Scaling**: Numerical columns ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare') are standardized to have a mean of 0 and a standard deviation of 1.

8. **Data Concatenation**: The processed numerical, label-encoded, and one-hot encoded data are concatenated along with the target variable to form the final dataset ready for modeling.

#### Model
I used LightGBM, CatBoost and decision tree models, and the final result was integrated by their prediction results.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: