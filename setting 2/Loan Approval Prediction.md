You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Loan_Approval_Prediction_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
OverviewWelcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: The goal for this competition is to predict whether an applicant is approved for a loan.StartOct 1, 2024CloseNov 1, 2024

##  Evaluation Metric:
Submissions are evaluated using area under the ROC curve using the predicted probabilities and the ground truth targets.
Submission File

For each id row in the test set, you must predict target loan_status. The file should contain a header and have the following format:

    id,loan_status
    58645,0.5
    58646,0.5
    58647,0.5
    etc.


##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Loan Approval Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Files

    train.csv - the training dataset; loan_status is the binary target
    test.csv - the test dataset; your objective is to predict probability of the target loan_status for each row
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length, loan_status
test.csv - column name: id, person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length


## Dataset folder Location: 
../../kaggle-data/playground-series-s4e10. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
Hey Kagglers! I used to be pretty active in these playground competitions, but after the December 2023 competition I took a break from Kaggle. On a whim I decided to start working on this one about 10 days ago, and it's been as much of a thrill as it ever was. Getting 1st place was a surprise, to be sure, but a welcome one!

### Cross-Validation
I'm sure you've heard this before, but setting up a robust cross-validation scheme for evaluating the performance of your predictions is VERY important to doing well in these competitions. I see lots of questions from folks on what kind of feature engineering to do, or how to best ensemble models, impute data, engineer features, etc. For a vast majority of these questions, there's no single answer that is universally true for any dataset. The only way to find out what works for a particular dataset is to try various options and see what performs the best, and that's where cross-validation comes in. In these playground competitions, the data is usually split 60-40 between train and test set, and 20% of the test set is used for the public leaderboard. That means that a CV score measures your performance on 60% of the entire dataset, whereas the public leaderboard measures your performance on only 8%, making cross-validation performance a much more reliable indicator of progress than public leaderboard performance. All of the decisions made below were based on optimizing my cross-validation performance.
### Data Preprocessing
Shoutout to various member of the community for the tip to treat the numerical features as categorical. What I found most effective was to maintain both the numeric feature and a categorical copy of it. I didn't do any other feature engineering, as my experience from past playground competitions has usually been that feature engineering is of little use. I did include the original dataset.
### Modelling
My general approach here is the same as the one I used last competition. For each of XGBoost, LightGBM, and CatBoost, I used Optuna to find 10 different sets of 'optimal hyperparameters' and averaged their predictions to get an overall prediction for each. Shoutout to @omidbaghchehsaraei's post here for the tip to use large max_bin values. I also added a Neural Network that was heavily inspired from @paddykb's notebook here. The performance of each of these models is as follows:

    Model CV Score Public LB Private LB
    LightGBM .96811 .97005 .96637
    XGBoost .96767 .96989 .96540
    CatBoost .96972 .97299 .96865
    NN .96678 .97088 .96577


What I think might have been my secret sauce was that for each of these model predictions, I trained a CatBoost model using the initial model predictions as a baseline. An example of how to do this can be found here. I'm not sure exactly what inspired me to do this, perhaps it was from seeing how amazingly well CatBoost performed on this data, but to my surprise CatBoost was able to significantly improve the performance of each of these model predictions, even the ones that were originally generated using CatBoost. The performance of these CatBoost-improved models are as follows:


Initial Model CV Score Public LB Private LB
LightGBM .96856 .97048 .96713
XGBoost .96815 .97024 .96611
CatBoost .96997 .97334 .96903
NN .96732 .97117 .96667

I find it impressive that the CatBoost model that used CatBoost predictions as a baseline would have been enough for 3rd place. CatBoost was the king for this comp! The final step was a Neural Network to stack these 4 predictions together. This squeezed out the extra last bit of performance needed to bring the solution to the top. 

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: