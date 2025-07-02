You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Binary_Classification_with_a_Bank_Churn_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the 2024 Kaggle Playground Series! Happy New Year! This is the 1st episode of Season 4. We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: For this Episode of the Series, your task is to predict whether a customer continues with their account or closes it (e.g., churns). Good luck!

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

**Submission File**
For each id in the test set, you must predict the probability for the target variable Exited. The file should contain a header and have the following format:

    id,Exited
    165034,0.9
    165035,0.1
    165036,0.5
    etc.

##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Bank Customer Churn Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance. Here we will predict a customer churn based on their information. The dataset contains 13 columns:

    Customer ID: A unique identifier for each customer
    Surname: The customer's surname or last name
    Credit Score: A numerical value representing the customer's credit score
    Geography: The country where the customer resides (France, Spain or Germany)
    Gender: The customer's gender (Male or Female)
    Age: The customer's age.
    Tenure: The number of years the customer has been with the bank
    Balance: The customer's account balance
    NumOfProducts: The number of bank products the customer uses (e.g., savings account, credit card)
    HasCrCard: Whether the customer has a credit card (1 = yes, 0 = no)
    IsActiveMember: Whether the customer is an active member (1 = yes, 0 = no)
    EstimatedSalary: The estimated salary of the customer
    Exited: Whether the customer has churned (1 = yes, 0 = no)

**Files**

    train.csv - the training dataset; Exited is the binary target
    test.csv - the test dataset; your objective is to predict the probability of Exited
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, Customer ID, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited
test.csv - column name: id, Customer ID, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary


## Dataset folder Location: 
../../kaggle-data/playground-series-s4e1. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
The following is the solution of Iqbal Syah Akbar, who is ranked at the 3rd position with the score of 0.9031.

### Feature Engineering

1. I applied TF-IDF vectorization on some features and then decompose the result with TruncatedSVD (credit to @arunklenin). However, unlike most people, I tried to build a class so I can implement it within a pipeline and do different type of vectorizations and decompositions on different set of features.
2. I created new features based on this notebook by @aspillai. However, scaling and IsSenior features aren't included. I also modified Sun_Geo_Gend_Sal for my own purposes. I also will refer to it as AllCat from now on.
3. I created a new feature named ZeroBalance as indicator whether a customer has zero balance or not.
4. Finally, I casted both EstimatedSalary and Age as integer by multiplying them by 100 and 10 respectively first. Why? You will see it the main reason soon. However, one funny side effect is that just multiplying the Age by 10 will give you a boost when paired with the code for binning from @aspillai notebook (it's equivalent of binning Age by dividing it by only 2).

### Encoding
Before we're getting into ensembling, I want to talk about encoding first, which is the key to getting high performance. There are three types of encoders I used in this competition: CatBoost's built-in encoder, CatBoost Encoder from category-encoders library, and M-Estimate Encoder. The first one is obvious why, but the second one is because I also wanted to use apply CatBoost encoding on other estimators. As for the third, it's because XGBoost and LightGBM doesn't like too much CatBoost encoding. I also won't explain much about it as it's not the main contributor to the high performance.

Now, let's get into what features I had encoded. Actually, let me rephrase the sentence. Let's get into what features I did not encode. In the original set of features, there are only Balance and HasCrCard as the unencoded features. The rest? Pretty much all encoded. This includes float features such as EstimatedSalary and Age, and now you know why I casted them as integer. Also, do you remember that I referred to one of feature engineering as AllCat? That's because I concatted almost all features I planned to encode in that feature, with exception of IsActiveMember because I also encoded IsActive_by_CreditCard. In total, there are 12 features I had encoded… or so you thought.

Remember TF-IDF vectorization and SVD decomposition? Well, you can encode them too! Just do the same thing as what I had done to EstimatedSalary and Age to the decomposition result. As a note, I only did encoding on SVD decomposition of Surname with 4 components, even though I also did vectorization and decomposition on AllCat and some other features.

Another important thing about the encoding here is that, CatBoost encoding actually cares about the order of your dataset so much. Well, not by default but, you can set it as such. In fact, category-encoders CatBoostEncoder treats different orders of data differently. In order for CatBoost to disallow permutation of dataset when encoding features, you have to set has_time parameter to True. And the best order of the dataset? When concatting the original dataset and the competition training dataset, you have to put original dataset before the competition dataset. This will give you the best result for this competition.

### Ensembling
I used 7 models on this competition.

1. Logistic Regression is the lowest performing model but also the greatest one for experimenting which features you need to encode with category-encoders CatBoostEncoder. In a way, this can function as indicator on which features you need to encode for CatBoost.
2. For Neural Network, I used Input -> 32 LeakyReLU -> 64 LeakyReLU -> 16 LeakyReLU -> 4 LeakyReLU -> 1 Sigmoid architecture, with AdamW optimizers. It has the same encoding as both Logistic Regression. Also, this is the only model where I didn't apply any vectorization.
3. XGBoost has both CatBoost Encoder and M-Estimate Encoder for different set of features, and I used Optuna for HPO. Vectorization is also applied to 4 features with 500 max features and 3 decomposition components.
4. LightGBM is somewhat similar to XGBoost when it comes to pre-processing.
5. 3 CatBoost with different bootstrap type each: no bootstrap, Bayesian, and Bernoulli, with same exact preprocessing: vectorization on 2 features with 1000 max features and 4 decomposition components as one of them. All of them have +0.902 CV score. I didn't do any HPO because they're already as slow as snail.

The weights are defined with Ridge Classifier. And if you're curious, I implemented all preprocessing within each model pipelines because I'm a freak when it comes to leakage preventation in cross-validation.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: