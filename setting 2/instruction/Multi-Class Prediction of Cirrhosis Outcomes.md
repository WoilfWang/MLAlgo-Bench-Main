You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Multi-Class_Prediction_of_Cirrhosis_Outcomes_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Your Goal: For this Episode of the Series, your task is to use a multi-class approach to predict the the outcomes of patients with cirrhosis. Good luck!

##  Evaluation Metric:
Submissions are evaluated using the multi-class logarithmic loss. Each id in the test set had a single true class label, Status. For each id, you must submit a set of predicted probabilities for each of the three possible outcomes, e.g., Status_C, Status_CL, and Status_D.
The metric is calculated
$$log loss = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}\log(p_{ij}),$$
where \(N\) is the number of rows in the test set, \(M\) is the number of outcomes (i.e., 3),  \(log\) is the natural logarithm, \(y_{ij}\) is 1 if row \(i\) has the ground truth label \(j\) and 0 otherwise, and \(p_{ij}\) is the predicted probability that observation \(i\) belongs to class \(j\).
The submitted probabilities for a given row are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, predicted probabilities are replaced with \(max(min(p,1-10^{-15}),10^{-15})\).

## Submission File
For each id row in the test set, you must predict probabilities of the three outcomes Status_C, Status_CL, and Status_D . The file should contain a header and have the following format:

    id,Status_C,Status_CL,Status_D
    7905,0.628084,0.034788,0.337128
    7906,0.628084,0.034788,0.337128
    7907,0.628084,0.034788,0.337128
    etc.

##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Cirrhosis Patient Survival Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.
Files

    train.csv - the training dataset; Status is the categorical target; C (censored) indicates the patient was alive at N_Days, CL indicates the patient was alive at N_Days due to liver a transplant, and D indicates the patient was deceased at N_Days.
    test.csv - the test dataset; your objective is to predict the probability of each of the three Status values, e.g., Status_C, Status_CL, Status_D.
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, N_Days, Drug, Age, Sex, Ascites, Hepatomegaly, Spiders, Edema, Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin, Stage, Status
test.csv - column name: id, N_Days, Drug, Age, Sex, Ascites, Hepatomegaly, Spiders, Edema, Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin, Stage


## Dataset folder Location: 
../../kaggle-data/playground-series-s3e26. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
The key was using a stacking approach with OOF from many other different top solutions and use XGB as meta model for training the stacked predictions + orginal features. And also try not to tune and optimize every solution to much, reduce the risk of overfitting, instead use the predictions as extra features in the final stacking training.

Many solutions shared the same small added feature engineering as below, rest was unchanged in terms of FE.

train_df['Age_Group'] = pd.cut(train_df['Age'], bins=[9000, 15000, 20000, 25000, 30000], labels=['A', 'B', 'C', 'D'],)
train_df['Log_Age'] = np.log1p(train_df['Age'])
scaler = MinMaxScaler()
train_df['Scaled_Age'] = scaler.fit_transform(train_df['Age'].values.reshape(-1, 1))

The different solutions and frameworks trained.

#### AutoGluon
AutoGluon latest pre-release 1.0.1b20231208 with zero-shot HPO as default. The trained framework used below weighted ensemble of models. I also tried distillation and pseudo labeling but didn’t work better.

    0 WeightedEnsemble_L2 -0.436142 log_loss 7.002513 505.384119 0.004281 9.923933 2 True 9
    1 XGBoost_r89_BAG_L1 -0.441355 log_loss 0.470275 19.330144 0.470275 19.330144 1 True 6
    2 CatBoost_r137_BAG_L1 -0.441618 log_loss 0.138598 115.856482 0.138598 115.856482 1 True 4
    3 CatBoost_r50_BAG_L1 -0.442764 log_loss 0.266119 78.536804 0.266119 78.536804 1 True 8
    4 LightGBM_r130_BAG_L1 -0.443168 log_loss 1.180046 49.676369 1.180046 49.676369 1 True 7
    5 XGBoost_r33_BAG_L1 -0.451191 log_loss 3.457726 68.160608 3.457726 68.160608 1 True 3
    6 RandomForestEntr_BAG_L1 -0.475263 log_loss 0.428620 3.049245 0.428620 3.049245 1 True 1
    7 NeuralNetTorch_r79_BAG_L1 -0.482747 log_loss 0.301830 66.300480 0.301830 66.300480 1 True 2
    8 NeuralNetFastAI_r145_BAG_L1 -0.489281 log_loss 0.755018 94.550054 0.755018 94.550054 1 True 5

#### LightAutoML
Trained the LightAutoML 0.3.8b1 version and the trained frameworked used below weighted model ensemble.

[16:19:15] Model description:

    Final prediction for new objects (level 0) =
    0.06558 * (5 averaged models Lvl_0_Pipe_0_Mod_0_LightGBM) +
    0.17057 * (5 averaged models Lvl_0_Pipe_0_Mod_1_Tuned_LightGBM) +
    0.27900 * (5 averaged models Lvl_0_Pipe_0_Mod_2_CatBoost) +
    0.48485 * (5 averaged models Lvl_0_Pipe_0_Mod_3_Tuned_CatBoost)

#### AutoXGB
Trained AutoXGB 5 fold CV with standard settings but with the extra features.









Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: