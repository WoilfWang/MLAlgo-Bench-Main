You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Explore_Multi-Label_Classification_with_an_Enzyme_Substrate_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the 2023 edition of Kaggle's Playground Series!

Thank you to everyone who participated in and contributed to Season 3 Playground Series so far! 

With the same goal to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science, we will continue launching the Tabular Tuesday in June every Tuesday 00:00 UTC, with each competition running for 2 weeks. Again, these will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc. 


Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the ground truth for each target, and the final score is the average of the individual AUCs of each predicted column. 

Submission File

For each id in the test set, you must predict the value for the targets EC1 and EC2. The file should contain a header and have the following format:

    id,EC1,EC2
    14838,0.22,0.71
    14839,0.78,0.43
    14840,0.53,0.11
    etc.


##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on a portion of the Multi-label Classification of enzyme substrates. This dataset only uses a subset of features from the original (the features that had the most signal). Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Note: For this challenge, you are given 6 features in the training data, but only asked to predict the first two features (EC1 and EC2).

Files

    train.csv - the training dataset; [EC1 - EC6] are the (binary) targets, although you are only asked to predict EC1 and EC2.
    test.csv - the test dataset; your objective is to predict the probability of the two targets EC1 and EC2
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3, HallKierAlpha, HeavyAtomMolWt, Kappa3, MaxAbsEStateIndex, MinEStateIndex, NumHeteroatoms, PEOE_VSA10, PEOE_VSA14, PEOE_VSA6, PEOE_VSA7, PEOE_VSA8, SMR_VSA10, SMR_VSA5, SlogP_VSA3, VSA_EState9, fr_COO, fr_COO2, EC1, EC2, EC3, EC4, EC5, EC6
test.csv - column name: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3, HallKierAlpha, HeavyAtomMolWt, Kappa3, MaxAbsEStateIndex, MinEStateIndex, NumHeteroatoms, PEOE_VSA10, PEOE_VSA14, PEOE_VSA6, PEOE_VSA7, PEOE_VSA8, SMR_VSA10, SMR_VSA5, SlogP_VSA3, VSA_EState9, fr_COO, fr_COO2


## Dataset folder Location: 
../../kaggle-data/playground-series-s3e18. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Sorry for publishing the solution a bit late.
Notebook can be found here: https://www.kaggle.com/code/nihilisticneuralnet/1st-place-winning-solution 

Importing Libraries: Importing essential libraries for data handling, preprocessing, model training, and evaluation. 
Loading Data: Loading the data

Data Preprocessing and Feature Engineering: Preprocesses the data by removing unnecessary columns ("id") and transforming the "mixed_desc" column by splitting its values. Additionally, new features are generated using groupby operations on categorical and numerical columns. These engineered features enhance the dataset for better model performance.

Defining Features and Targets: The features and target variables are established, separating the input data for training purposes. The additional features are generated by combining categorical and numerical columns, creating a more informative dataset.

Model Initialization and Parameters: Configuration parameters for two classifiers, XGBoost and LightGBM, are defined. These parameters determine how the models learn from the data. The MultiOutputClassifier wrapper is employed to handle multi-output prediction tasks.

Creating Pipelines: Two pipelines are set up for training using the defined classifiers. These pipelines streamline the process of training and prediction, encapsulating the necessary steps within each model.

Training and Validation Loop: Cross-validation is performed using RepeatedMultilabelStratifiedKFold. Then iterating through the specified number of folds, training the classifiers on training data and evaluating their performance on validation data. Predictions are generated, and ROC AUC scores are calculated to gauge how well the models perform.

Performance Evaluation and Averaging: The average ROC AUC scores for each classifier are calculated over all folds, both for training and validation data. An "overall" prediction set is created by averaging predictions from both classifiers, aiding in robust performance assessment.

Test Set Predictions: Predictions are generated for the test dataset using the trained classifiers. The predictions from XGBoost and LightGBM are averaged to form the final predictions for the test set.

Submission: And at last, submitting to the competition..


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: