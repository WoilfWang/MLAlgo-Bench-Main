You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Multi-Class_Prediction_of_Obesity_Risk_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: The goal of this competition is to use various factors to predict obesity risk in individuals, which is related to cardiovascular disease. Good luck!

##  Evaluation Metric:
Submissions are evaluated using the accuracy score.

Submission File
For each id row in the test set, you must predict the class value of the target, NObeyesdad. The file should contain a header and have the following format:

    id,NObeyesdad
    20758,Normal_Weight
    20759,Normal_Weight
    20760,Normal_Weight
    etc.

##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Obesity or CVD risk dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Note: This dataset is particularly well suited for visualizations, clustering, and general EDA. Show off your skills!

Files

    train.csv - the training dataset; NObeyesdad is the categorical target
    test.csv - the test dataset; your objective is to predict the class of NObeyesdad for each row
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS, NObeyesdad
test.csv - column name: id, Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS


## Dataset folder Location: 
../../kaggle-data/playground-series-s4e2. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:

#### Summary
A stacking approach with XGB as meta learning and different SOTA solutions as extra added stacking features from each prediction. Final inference includes optimized accuracy metric and 9 predictions of different version of the stacking code and other diverse solutions for a count of max class per row.

#### Data and Feature Engineering
Both the competition dataset and the extra dataset was used.

I used two different FE for training:

**FE 1**


1.	Concatenation and Cleaning:

        •	The train_df and original DataFrames are concatenated vertically using pd.concat(), combining both datasets.
        •	The id column is dropped from the concatenated dataset using drop(['id'], axis=1).
        •	Any duplicate rows are removed using .drop_duplicates(), and the index is reset using .reset_index(drop=True) to ensure the data is clean and properly indexed.
2.	Feature Engineering for Training Data:

        •	Age Grouping: A new column, Age_Group, is created by binning the Age column into categories (A, B, C, D) for different age ranges (20-30, 30-40, 40-50, 50-55) using pd.cut().
        •	Log Transformation of Age: A new column Log_Age is created by applying a logarithmic transformation (np.log1p()) to the Age column. This transformation helps in reducing skewness if the age distribution is highly skewed.
        •	Scaling of Age: The Age values are scaled between 0 and 1 using MinMaxScaler(). The scaler is fitted to the Age column, and the transformed values are stored in the new column Scaled_Age. This normalization helps ensure that the model doesn’t favor features with larger scales.
3.	Test Data Processing:

        •	The test dataset (test.csv) is loaded into test_df.
        •	Similar feature engineering steps are applied to the test data:
        •	Age Grouping: The Age values are categorized into the same bins as in the training set.
        •	Log Transformation of Age: The Log_Age transformation is applied to the test dataset’s Age column.
        •	Scaling of Age: The test dataset’s Age values are scaled using the same scaler that was fitted on the training data (scaler.transform()).
        •	The id column is dropped from the test data.
4.	Label Encoding:
        •	The target variable NObeyesdad (which represents obesity categories) in the training data is label encoded. This transforms the categorical values into numeric labels, making it suitable for machine learning algorithms that expect numerical targets. The LabelEncoder is fit on the unique values of NObeyesdad, and then it is used to transform the NObeyesdad column in the training data.

In summary, the code processes both training and test datasets by:
	•	Merging and cleaning the data.
	•	Engineering new features (like age group, log-transformed age, and scaled age).
	•	Label encoding the target variable for classification tasks.
This preprocessing ensures that the datasets are ready for training machine learning models.

And

**FE 2**
1.	Age Group Creation:

        •	For the training (train), test (test), and original (original) datasets, the Age column is used to create a new categorical feature called Age group.
        •	The pd.cut() function is used to bin the Age values into five distinct age groups:
        •	0-18
        •	19-30
        •	31-45
        •	46-60
        •	60+
        •	The bin edges are dynamically determined using the maximum value in each dataset’s Age column (train['Age'].max(), test['Age'].max(), original['Age'].max()), ensuring that the final group covers all ages in the dataset.
2.	BMI Calculation:

        •	A new feature called BMI (Body Mass Index) is calculated for the train, test, and original datasets. The formula used is:

        $$BMI = \frac{{\text{{Weight}}}}{{\text{{Height}}^2}}$$

	    •	This formula computes the BMI based on the Weight and Height columns in each dataset.
3.	Interaction Feature (Age * Gender):

	    •	A new feature, Age * Gender, is created by multiplying the Age and Gender columns for the train, test, and original datasets. This interaction feature may capture the combined effect of age and gender on the target variable.
4.	One-Hot Encoding of Categorical Features:

	    •	The datasets (train, test, and original) contain several categorical features, such as Gender, family_history_with_overweight, Age group, and others.
	    •	These categorical features are transformed into numerical representations using one-hot encoding with pd.get_dummies(). This process creates binary columns for each category within each feature, allowing the model to interpret them as numerical data.
5.	Polynomial Feature Engineering:

	    •	Polynomial features are generated for the numerical features Age and BMI using the PolynomialFeatures class from sklearn.preprocessing.
	    •	The degree of the polynomial is set to 2, meaning that the following new features are generated for both train, test, and original datasets:
	    •	Age^2: Square of the Age feature.
	    •	Age^3: Cube of the Age feature.
	    •	BMI^2: Square of the BMI feature.
	    •	Interaction terms like Age * BMI, Age * BMI^2, and Age * BMI^3 capture the combined effects of Age and BMI in polynomial relationships.
	    •	The generated polynomial features are concatenated with the original datasets, creating new columns that represent more complex relationships between the variables.
6.	Concatenation of Polynomial Features:

	    •	The polynomial features are added to the respective datasets (train, test, original) using pd.concat(), with the newly created columns being named appropriately (e.g., Age^2, Age^3, etc.).

Summary of the Data Processing Pipeline:
	•	Age Grouping: Categorizes Age into predefined groups.
	•	BMI Calculation: Adds a new feature for BMI.
	•	Interaction Feature: Creates a new feature by multiplying Age and Gender.
	•	One-Hot Encoding: Converts categorical variables into binary features.
	•	Polynomial Features: Generates additional features by applying polynomial transformations to Age and BMI, and their interactions.


#### Models and frameworks used to the stacking approach.

Competition metric is Accuracy but for training log_loss was set and probability per solution was saved for later use.

AutoXGB

AutoGluon with the new zero-shot HPO training.
Ensemble Weights: {'CatBoost_r9_BAG_L1': 0.363, 'LightGBM_r131_BAG_L1': 0.253, 'XGBoost_BAG_L1': 0.099, 'XGBoost_r33_BAG_L1': 0.099, 'NeuralNetTorch_BAG_L2': 0.077, 'ExtraTreesEntr_BAG_L2': 0.033, 'NeuralNetFastAI_BAG_L2': 0.022, 'CatBoost_BAG_L2': 0.022, 'NeuralNetTorch_BAG_L1': 0.011, 'RandomForestGini_BAG_L2': 0.011, 'ExtraTreesGini_BAG_L2': 0.011}

LightAutoml with LGBM and Catboost.

Custom XGB + LGBM training code with and without Pseudo Label training.

#### Stacking

Used the features from training and added the probability from the different solution as extra features. The meta model was XGB. Stacking is a great approach in classification problem versus other techniques.



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: